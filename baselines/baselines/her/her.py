import numpy as np

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return -np.linalg.norm(goal_a - goal_b, axis=-1)

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info


        transitions['r'] = reward_fun(**reward_params)
        
        loop_size = transitions['ag_2'].shape[0]

        row_remove = []

        for k in range(loop_size):
            distance = goal_distance (transitions['ag_2'][k], transitions['g'][k])

            # Remove this transition
            if (distance <= -0.4):
                row_remove.append(k)

        if len(row_remove) > 0:
            # Remove all low-rank experiences
            # transitions
            for keys in transitions.keys():
                transitions[keys] = np.delete(transitions[keys], (row_remove), axis=0)
            # print(distance)
        # print(loop_size)

        # for k in ['ag_2', 'g']:        
        #     distance = goal_distance (transitions['ag_2'], transitions['g'])

        # print(transitions['info_is_rank'].shape)

        # for dis in range(len(distance)):
        #     if distance[dis] <= -0.

        # print(distance.shape)

        batch_size = transitions['ag_2'].shape[0]

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        # assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
