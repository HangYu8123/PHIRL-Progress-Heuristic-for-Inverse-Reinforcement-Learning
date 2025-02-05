# A modified version of airl.py from imitation library
# add reward shaping
"""Adversarial Inverse Reinforcement Learning (AIRL)."""
from typing import Callable, Iterable, Iterator, Mapping, Optional, Type, overload

import torch as th
from torch.nn import functional as F

from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.common import compute_train_stats
from imitation.rewards import reward_nets

from imitation.data import types
from copy import deepcopy
import numpy as np
import imitation.scripts.train_adversarial as train_adversarial
import os
from pathlib import Path

STOCHASTIC_POLICIES = (sac_policies.SACPolicy, policies.ActorCriticPolicy)


class PHIRL(common.AdversarialTrainer):
    """Adversarial Inverse Reinforcement Learning (`AIRL`_).

    .. _AIRL: https://arxiv.org/abs/1710.11248
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        annotation_list: list[tuple[int, dict]],
        demostrations_for_shaping: list[types.Trajectory],
        shape_reward = [],
        shaping_batch_size: int = 16,
        shaping_loss_weight: float = 1.0,
        shaping_update_freq: int = 1,
        shaping_lr: float = 1e-3,
        save_model_every = 10,
        save_path = "checkpoints/default",
        traj_index = [], 
        **kwargs,
    ):
        """Builds an AIRL trainer.

        Args:
            annotation_list: list of annotation tuples: (dict(progress_data), int(corresponding demonstration index))
            shape_reward: a list of reward shapings to use, no reward shaping if empty, can contain: ["progress_sign_loss"]
        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute (present in `ActorCriticPolicy`), needed to compute
                log-probability of actions.
        """
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )
        # AIRL needs a policy from STOCHASTIC_POLICIES to compute discriminator output.
        if not isinstance(self.gen_algo.policy, STOCHASTIC_POLICIES):
            raise TypeError(
                "AIRL needs a stochastic policy to compute the discriminator output.",
            )
        
        assert isinstance(demostrations_for_shaping, list), "demonstrations_for_shaping must be a list of Trajectory"
        assert isinstance(demostrations_for_shaping[0], types.Trajectory), "demonstrations_for_shaping must be a list of Trajectory"
        assert isinstance(annotation_list, list), "annotation_dict must be a list"

        self.demonstrations = deepcopy(demonstrations)
        self.demonstrations_for_shaping = deepcopy(demostrations_for_shaping)
        self.annotation_list = annotation_list
        self.shaping_batch_size = shaping_batch_size
        self.shaping_loss_weight = shaping_loss_weight
        self.shaping_update_freq = shaping_update_freq
        self.shaping_lr = shaping_lr
        self.shape_reward = shape_reward
        self.save_model_every = save_model_every
        self.save_path = save_path 
        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        self.save_path = os.path.join(self.project_path, self.save_path)
        self.traj_index = traj_index
        if not os.path.exists(self.save_path):
            print("creating save path")
            os.makedirs(self.save_path)
        #print("n_disc per round", self.n_disc_updates_per_round)
        # path from str to Path
        # self.save_path = Path(self.save_path)

    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        r"""Compute the discriminator's logits for each state-action sample.

        In Fu's AIRL paper (https://arxiv.org/pdf/1710.11248.pdf), the
        discriminator output was given as

        .. math::

            D_{\theta}(s,a) =
            \frac{ \exp{r_{\theta}(s,a)} } { \exp{r_{\theta}(s,a)} + \pi(a|s) }

        with a high value corresponding to the expert and a low value corresponding to
        the generator.

        In other words, the discriminator output is the probability that the action is
        taken by the expert rather than the generator.

        The logit of the above is given as

        .. math::

            \operatorname{logit}(D_{\theta}(s,a)) = r_{\theta}(s,a) - \log{ \pi(a|s) }

        which is what is returned by this function.

        Args:
            state: The state of the environment at the time of the action.
            action: The action taken by the expert or generator.
            next_state: The state of the environment after the action.
            done: whether a `terminal state` (as defined under the MDP of the task) has
                been reached.
            log_policy_act_prob: The log probability of the action taken by the
                generator, :math:`\log{ \pi(a|s) }`.

        Returns:
            The logits of the discriminator for each state-action sample.

        Raises:
            TypeError: If `log_policy_act_prob` is None.
        """
        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )
        reward_output_train = self._reward_net(state, action, next_state, done)
        return reward_output_train - log_policy_act_prob

    def progress_shaping_loss(self) -> th.Tensor:

        '''
        get progress from annotations and compute the progress shaping loss
        '''
        # randomly choose some demonstrations from annotation_list
        # randomly generate a batch of indices
        indices = np.random.choice(len(self.annotation_list), self.shaping_batch_size , replace=False)
        threshold = 0.1
        # get corresponding annotations
        annotations = [self.annotation_list[idx] for idx in indices]
        #print("indices", indices)
        # get the progress change from annotations
        # print("type check",type(annotations[0][0]["start_progress"]))
        delta_progress = th.tensor([annotation[0]["end_progress" ]  - annotation[0]["start_progress"] -  threshold for annotation in annotations])
        #print("delta_progress", delta_progress)
        # print("what is this",annotations[0][0])
        #progress = th.tensor([annotation[0]["end_progress"] for annotation in annotations])

        # get the average progress value
        average_progress_value = th.tensor(([(annotation[0]["start_progress"] + annotation[0]["end_progress"]) / 2 for annotation in annotations]))

        # get corresponding states and actions
        demostration_indicies = [(annotation[1], annotation[0]["start_step"], annotation[0]["end_step"]) for annotation in annotations]
        states = [th.tensor(self.demonstrations_for_shaping[demostration_index].obs[start_step:end_step], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        actions = [th.tensor(self.demonstrations_for_shaping[demostration_index].acts[start_step:end_step], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        
        next_states = [th.tensor(self.demonstrations_for_shaping[demostration_index].obs[start_step+1:end_step+1], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        dones = [int(self.demonstrations_for_shaping[demostration_index].terminal) for demostration_index, _, end_step in demostration_indicies]

        # record corresponding index for each part of the batch
        state_lengths = th.tensor([len(state) for state in states])
        # add 0 to the beginning of the tensor
        state_lengths = th.cat((th.tensor([0]), state_lengths))

        # accumulate the lengths of the states
        state_indicies = th.cumsum(state_lengths, dim=0)
        
        # concatenate the states and actions
        states = th.cat(states)
        actions = th.cat(actions)
        next_states = th.cat(next_states)
        dones = th.tensor(dones, dtype=th.float32)

        # length check
        if len(states)!= len(next_states) or len(states) != len(actions):
            # reduce the length of states by the minimum length
            min_length = min(len(states), len(next_states), len(actions))
            states = states[:min_length]
            actions = actions[:min_length]
            next_states = next_states[:min_length]
            dones = dones[:min_length]
    

        # to device
        states = states.to(self.gen_algo.device)
        actions = actions.to(self.gen_algo.device)
        next_states = next_states.to(self.gen_algo.device)
        dones = dones.to(self.gen_algo.device)

        # get the reward output from the reward network
        
        reward_output_train = self._reward_net.base(states, actions, next_states, dones)
        v_s = self._reward_net.potential(states)
        v_s_next = self._reward_net.potential(next_states)
        delta_value = v_s_next - v_s
        advatanage_output = th.tensor([0.0 for i in range(len(states))])
        advatanage_output = self._reward_net(states, actions, next_states, th.tensor(1))


        # get the value output from the potential part of reward network
        old_value_output_train = self._reward_net.potential(states).flatten()
        next_value_output_train = self._reward_net.potential(next_states).flatten()


        # sum the reward output for each trajectory
        reward_output_train = th.stack([reward_output_train[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])

        delta_value = th.stack([delta_value[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])
        advatanage_output = th.stack([advatanage_output[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)]) 
        

        # sum the value output for each trajectory
        old_value_output_train = th.stack([old_value_output_train[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])
        next_value_output_train = th.stack([next_value_output_train[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])
        
        # the reward sum should have same length as delta_progress
        assert len(reward_output_train) == len(delta_progress), "reward_output_train and delta_progress should have same length"
        assert len(old_value_output_train) == len(delta_progress), "old_value_output_train and delta_progress should have same length"
        assert len(next_value_output_train) == len(delta_progress), "next_value_output_train and delta_progress should have same length"

        # to device
        delta_progress = delta_progress.to(self.gen_algo.device)
        average_progress_value = average_progress_value.to(self.gen_algo.device)
        reward_output_train = reward_output_train.to(self.gen_algo.device)
        old_value_output_train = old_value_output_train.to(self.gen_algo.device)
        next_value_output_train = next_value_output_train.to(self.gen_algo.device)

        #loss_sign = self.progress_sign_loss(delta_progress, reward_output_train)
        loss_scale = self.delta_progress_reward_loss(delta_progress, reward_output_train)
        loss_value = self.value_sign_loss(delta_progress, delta_value)
        loss_advantage = self.advantage_sign_loss(delta_progress, advatanage_output)
        loss_progress_reward= self.reward_sign_loss(average_progress_value, next_value_output_train)
        if "subtrajectory_proportion_loss" in self.shape_reward:
            loss_proportion = self.subtrajectory_proportion_loss()
        if "end_progress_loss" in self.shape_reward:
            loss_end_progress = self.end_progress_loss()

        # return the loss
        return_dict = {"delta_progress_reward_loss": loss_scale,
                        "value_sign_loss": loss_value,
                        "advantage_sign_loss": loss_advantage,
                        "reward_sign_loss": loss_progress_reward,
                        #"subtrajectory_proportion_loss": loss_proportion,
                        #"end_progress_loss": loss_end_progress
                        }
        if "end_progress_loss" in self.shape_reward:
            return_dict["end_progress_loss"] = loss_end_progress
        if "subtrajectory_proportion_loss" in self.shape_reward:
            return_dict["subtrajectory_proportion_loss"] = loss_proportion

        return return_dict

    def advantage_sign_loss(self,
                            delta_progress: th.Tensor,
                            delta_advantage: th.Tensor) -> th.Tensor:
        """
        Enforce that the (differentiable) signs of the progress change and the advantage change match.
        We compute the soft-sign of each quantity and use an MSE loss.
        """
        device = self.gen_algo.device
        loss = F.mse_loss(F.softsign(delta_advantage).to(device),
                        F.softsign(delta_progress).to(device))
        return loss
    def reward_sign_loss(self, 
                        delta_progress: th.Tensor, 
                        reward_output_train: th.Tensor) -> th.Tensor:
        """
        Enforce that the (differentiable) sign of the reward matches that of the progress change.
        We use MSE loss on the soft-sign values.
        """
        device = self.gen_algo.device
        loss = F.mse_loss(F.softsign(reward_output_train).to(device),
                        F.softsign(delta_progress).to(device))
        return loss
    def value_sign_loss(self,
                        delta_progress: th.Tensor,
                        delta_value: th.Tensor) -> th.Tensor:
        """
        Enforce that the (differentiable) sign of the value difference matches that of the progress difference.
        We use MSE loss on the soft-sign values.
        """
        device = self.gen_algo.device
        loss = F.mse_loss(F.softsign(delta_value).to(device),
                        F.softsign(delta_progress).to(device))
        return loss


    def delta_progress_reward_loss(self, 
                                 delta_progress: th.Tensor, 
                                 reward_output_train: th.Tensor) -> th.Tensor:
        """
        For a single demonstration’s subtrajectories, enforce that the difference in reward
        between any two subtrajectories matches the difference in progress.
        This function computes pairwise differences (only within one demonstration) and applies an MSE loss.
        """
        device = self.gen_algo.device
        if delta_progress.numel() < 2:
            return th.tensor(0.0, device=device)
        # Compute pairwise differences (resulting in square matrices)
        diff_progress = delta_progress.unsqueeze(1) - delta_progress.unsqueeze(0)
        diff_reward = reward_output_train.unsqueeze(1) - reward_output_train.unsqueeze(0)
        loss = F.mse_loss(diff_reward, diff_progress)
        return loss
    def end_progress_loss(self) -> th.Tensor:
        """
        For randomly sampled pairs of demonstrations, compare their final progress and total reward.
        If the difference in final progress is small (less than 10), we want the difference in total rewards
        to be small (less than 10% of the larger reward). If the progress difference is large, we require that
        the reward difference be at least 5% of the larger reward.
        """
        device = self.gen_algo.device
        if len(self.demonstrations_for_shaping) < 2 or len(self.traj_index) < 2:
            return th.tensor(0.0, device=device)
        
        num_pairs = 4
        # Sample distinct demonstration indices (2*num_pairs items)
        indices = np.random.choice(self.traj_index, size=2 * num_pairs, replace=False)
        losses = []
        for k in range(num_pairs):
            i = indices[2*k]
            j = indices[2*k+1]
            # Get final progress for demonstration i and j from annotations.
            ann_i = [ann for ann in self.annotation_list if ann[1] == i]
            if not ann_i:
                continue
            final_progress_i = max(ann[0]["end_progress"] for ann in ann_i)
            ann_j = [ann for ann in self.annotation_list if ann[1] == j]
            if not ann_j:
                continue
            final_progress_j = max(ann[0]["end_progress"] for ann in ann_j)
            
            # Compute total reward for demonstration i.
            demo_i = self.demonstrations_for_shaping[i]
            states_i = th.tensor(demo_i.obs, dtype=th.float32, device=device)
            acts_i = th.tensor(demo_i.acts, dtype=th.float32, device=device)
            next_i = th.tensor(demo_i.obs, dtype=th.float32, device=device)
            dones_i = th.zeros(len(states_i), dtype=th.float32, device=device)
            reward_i = self._reward_net.base(states_i, acts_i, next_i, dones_i).sum()
            
            # Compute total reward for demonstration j.
            demo_j = self.demonstrations_for_shaping[j]
            states_j = th.tensor(demo_j.obs, dtype=th.float32, device=device)
            acts_j = th.tensor(demo_j.acts, dtype=th.float32, device=device)
            next_j = th.tensor(demo_j.obs, dtype=th.float32, device=device)
            dones_j = th.zeros(len(states_j), dtype=th.float32, device=device)
            reward_j = self._reward_net.base(states_j, acts_j, next_j, dones_j).sum()
            
            diff_progress = abs(final_progress_i - final_progress_j)
            diff_reward = th.abs(reward_i - reward_j)
            reward_max = th.max(reward_i, reward_j).clamp_min(1e-6)
            
            if diff_progress < 10.0:
                # For similar progress, penalise if rewards differ by more than 10% of the maximum reward.
                loss_pair = th.clamp(diff_reward - 0.1 * reward_max, min=0) / reward_max
            else:
                # For dissimilar progress, penalise if rewards differ by less than 5% of the maximum reward.
                loss_pair = th.clamp(0.05 * reward_max - diff_reward, min=0) / reward_max
            
            losses.append(loss_pair)
        
        if len(losses) == 0:
            return th.tensor(0.0, device=device)
        return th.mean(th.stack(losses))

    def subtrajectory_proportion_loss(self) -> th.Tensor:
        """
        For each demonstration, if it has annotated segmentation points, then at each annotation
        (which records an "end_progress" and an "end_step"), compute:
        - The proportion of progress achieved (end_progress / final_progress)
        - The proportion of cumulative reward (cumulative reward up to that step / total reward)
        and penalize their difference via an L1 loss.
        """
        device = self.gen_algo.device
        from collections import defaultdict
        # Group annotations by demonstration index.
        demo_annotations = defaultdict(list)
        for ann in self.annotation_list:
            demo_annotations[ann[1]].append(ann[0])
        
        losses = []
        for demo_idx, anns in demo_annotations.items():
            if len(anns) < 1:
                continue
            # Sort annotations by end_step.
            anns.sort(key=lambda x: x["end_step"])
            demo = self.demonstrations_for_shaping[demo_idx]
            states = th.tensor(demo.obs, dtype=th.float32, device=device)
            actions = th.tensor(demo.acts, dtype=th.float32, device=device)
            next_states = th.tensor(demo.obs, dtype=th.float32, device=device)
            dones = th.zeros(len(states), dtype=th.float32, device=device)
            all_rewards = self._reward_net.base(states, actions, next_states, dones)
            total_reward = all_rewards.sum()
            # Use the final annotation's progress as final progress.
            final_progress = anns[-1]["end_progress"]
            if abs(final_progress) < 1e-6 or total_reward.item() == 0:
                continue
            demo_losses = []
            for ann in anns:
                step = min(ann["end_step"], len(all_rewards))
                cum_reward = all_rewards[:step].sum()
                prop_reward = cum_reward / (total_reward + 1e-8)
                prop_progress = ann["end_progress"] / final_progress
                demo_losses.append(th.abs(prop_reward - prop_progress))
            if demo_losses:
                losses.append(th.mean(th.stack(demo_losses)))
        
        if len(losses) == 0:
            return th.tensor(0.0, device=device)
        return th.mean(th.stack(losses))

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._reward_net
        # Recursively return the base network of the wrapped reward net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net


    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        
        '''
        a modified version of train_disc from airl.py with the following changes:
        - add one more training step for the discriminator to shape the reward
        '''


        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
        """

        

        # original training code
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0
            self._disc_opt.zero_grad()



            # compute loss
            

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )

            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )

                # Renormalise the loss to be averaged over the whole
                # batch size instead of the minibatch size.
                #print("loss before:", loss)
                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                
                if len(self.shape_reward) > 0 and self._disc_step % self.shaping_update_freq == 0:
                #self._disc_opt.zero_grad()
                    shaping_losses = self.progress_shaping_loss()
                    # get the losses in self.shape_reward list using keys, sum them
                    shaping_loss = sum([shaping_losses[key] for key in self.shape_reward])
                    print("************************************************************")
                    for key in self.shape_reward:
                        print(key, shaping_losses[key])
                    print("AIRL loss:", loss)
                    print("************************************************************")
                else:
                    shaping_loss = th.tensor(0.0, device=self.gen_algo.device)

                combined_loss = loss + shaping_loss
                combined_loss.backward()
                #loss.backward()

            # do gradient step
            self._disc_opt.step()
            self._disc_step += 1

            # reward shaping loss

                # relase unused loss
                # del shaping_losses


                # if "progress_sign_loss" in self.shape_reward and self._disc_step % self.shaping_update_freq == 0:
                #     self._disc_opt.zero_grad()
                #     shaping_loss = self.progress_shaping_loss()
                #     print(shaping_loss)
                #     shaping_loss *= self.shaping_loss_weight
                #     shaping_loss.backward()
                #     self._disc_opt.step()

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())
        # save the model
        if self._global_step % self.save_model_every == 0:
            # set save path according to the global step
            save_path_this_time = os.path.join(self.save_path, f"{self._global_step}")
            if not os.path.exists(save_path_this_time):
                os.makedirs(save_path_this_time)
            save_path_this_time = Path(save_path_this_time)
            train_adversarial.save(self, 
                                   save_path_this_time,)
        return train_stats
    
