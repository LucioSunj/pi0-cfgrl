from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor


class HGCBCAgent(flax.struct.PyTreeNode):
    """Hierarchical goal-conditioned behavioral cloning (HGCBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level BC actor loss."""
        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['high_actor_actions'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['high_actor_actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info

    def low_actor_loss(self, batch, grad_params, rng=None):
        """Compute the low-level BC actor loss."""
        dist = self.network.select('low_actor')(batch['observations'], batch['low_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        rng, low_actor_rng = jax.random.split(rng)
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = high_actor_loss + low_actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        subgoals = high_dist.sample(seed=high_seed)

        low_dist = self.network.select('low_actor')(observations, subgoals, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['high_actor_goals']
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]

        # Define encoder.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            raise NotImplementedError

        # Define actor networks.
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=goal_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=encoders.get('high_actor'),
        )

        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=encoders.get('low_actor'),
        )

        network_info = dict(
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
            low_actor=(
                low_actor_def,
                (ex_observations, ex_goals),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='hgcbc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            subgoal_steps=25,  # Subgoal steps.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
