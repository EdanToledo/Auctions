import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Tuple
from jumanji.env import Environment
from jumanji.specs import Spec, Array
from jumanji.types import TimeStep, restart, transition

class AuctionState(NamedTuple):
    key: jax.Array              # Random key for reproducibility
    valuations: jax.Array       # Agents' private valuations, shape (num_agents,)
    cumulative_utility: jax.Array  # Cumulative utility, shape (num_agents,)
    step_no: int                # Current step number in the auction

class Auction(Environment):
    def __init__(self, num_agents: int, num_rounds: int, max_valuation: float) -> None:
        super().__init__()
        self.num_agents = num_agents           # Number of agents in the auction
        self.num_rounds = num_rounds           # Total number of auction rounds
        self.max_valuation = max_valuation     # Maximum possible valuation for agents

    def reset(self, key: jax.Array) -> Tuple[AuctionState, TimeStep]:
        """
        Resets the environment to an initial state.

        Args:
            key: JAX random key for random number generation.

        Returns:
            state: Initial environment state.
            timestep: TimeStep object containing the first dummy observation and the private agent valuations.
        """
        # Split the random key for reproducibility
        key, subkey = random.split(key)
        # Sample private valuations uniformly between 0 and max_valuation for all agents
        valuations = random.uniform(subkey, minval=0.0, maxval=self.max_valuation, shape=(self.num_agents,))
        # Initialize cumulative utility to zero for all agents
        state = AuctionState(
            key=key,
            valuations=valuations,
            cumulative_utility=jnp.zeros_like(valuations),
            step_no=0
        )
        # Initial dummy observation (agent indices, winner index and winner bid)
        agent_ids = jax.nn.one_hot(jnp.arange(self.num_agents), self.num_agents)
        winning_id = jnp.int32(-1).repeat(self.num_agents).reshape(self.num_agents, 1)
        winning_bid = jnp.float32(-1).repeat(self.num_agents).reshape(self.num_agents, 1)
        observation = jnp.concatenate([agent_ids, winning_id, winning_bid], axis=-1)
        # Include private valuations in the extras field
        extras = {"valuations": state.valuations}
        # Create the initial TimeStep
        timestep = restart(observation, shape=(self.num_agents,), extras=extras)
        return state, timestep

    def step(self, state: AuctionState, bids: jax.Array) -> Tuple[AuctionState, TimeStep]:
        """
        Executes one time step within the environment.

        Args:
            state: The current state of the environment.
            bids: Agents' bids, shape (num_agents,).

        Returns:
            next_state: The updated state after executing the step.
            timestep: TimeStep object containing rewards, observations, and other info.
        """
        key = state.key
        valuations = state.valuations  # Agents' private valuations

        # Determine the highest bid among all agents
        max_bid = jnp.max(bids)
        # Identify the index of the agent with the highest bid (ties go to the agent with the lowest index)
        winner_index = jnp.argmax(bids == max_bid)

        # Create an array of agent indices
        agent_indices = jnp.arange(valuations.shape[0])
        # Determine which agent won
        agent_won = agent_indices == winner_index  # Boolean array indicating the winning agent
        # Compute rewards: winning agent gets (valuation - bid), others get zero
        rewards = jnp.where(agent_won, valuations - bids, jnp.zeros_like(bids))

        # Update cumulative utility
        new_cumulative_utility = state.cumulative_utility + rewards

        # Check if the episode is finished
        done = jax.lax.select(state.step_no == self.num_rounds - 1, jnp.bool_(True), jnp.bool_(False))

        # Split the key for the next round
        key, subkey = random.split(key)

        # Prepare the next state
        next_state = state._replace(
            key=key,
            cumulative_utility=new_cumulative_utility,
            step_no=state.step_no + 1
        )

        # Prepare the observation for the next step
        agent_ids = jax.nn.one_hot(jnp.arange(self.num_agents), self.num_agents)
        winning_id = winner_index.repeat(self.num_agents).reshape(self.num_agents, 1)
        winning_bid = max_bid.repeat(self.num_agents).reshape(self.num_agents, 1)
        new_observation = jnp.concatenate([agent_ids, winning_id, winning_bid], axis=-1)

        # Compute the discount factor
        discount = jnp.ones_like(rewards) * (1 - done)
        # Create the TimeStep object
        timestep = transition(
            reward=rewards,
            observation=new_observation,
            discount=discount,
            extras={"valuations": valuations}
        )

        return next_state, timestep

    def observation_spec(self) -> Spec:
        return Array((self.num_agents, self.num_agents + 2), dtype=jnp.float32, name="observation")
    
    def action_spec(self) -> Spec:
        return Array((self.num_agents,), dtype=jnp.float32, name="bids")
    
    def render(self, state: AuctionState, bids: jax.Array = None) -> None:
        """
        Renders the current state of the auction. Just for pretty printing.

        Args:
            state: The current state of the environment.
            bids: (Optional) The bids submitted by the agents in the current step.
        """
        valuations = jnp.round(state.valuations, 2)
        cumulative_utility = jnp.round(state.cumulative_utility, 2)
        step_no = state.step_no

        print(f"\n=== Auction Round {step_no} ===")
        print("Agent Valuations:")
        for i, val in enumerate(valuations):
            print(f"  Agent {i}: Valuation = {val}")

        if bids is not None:
            bids = jnp.round(bids, 2)
            # Determine the highest bid and the winner
            max_bid = jnp.max(bids)
            winner_index = jnp.argmax(bids == max_bid)

            print("\nAgents' Bids:")
            for i, bid in enumerate(bids):
                print(f"  Agent {i}: Bid = {bid}")

            print(f"\nWinning Agent: Agent {winner_index} with a bid of {max_bid}")
            print("\nCumulative Utilities:")
            for i, util in enumerate(cumulative_utility):
                print(f"  Agent {i}: Cumulative Utility = {util}")
        else:
            print("\nNo bids have been submitted yet.")

        print("========================\n")
        
        
if __name__ == "__main__":
    
    def run_auction():
        # Initialize random key
        key = jax.random.PRNGKey(0)
        # Create an Auction environment
        env = Auction(num_agents=3, num_rounds=2, max_valuation=10.0)
        
        # Reset the environment
        state, timestep = env.reset(key)
        
        # Render the initial state
        env.render(state)
        
        # Agents submit bids for the first round
        bids_round_1 = jnp.array([5.0, 7.0, 6.0])
        state, timestep = env.step(state, bids_round_1)
        env.render(state, bids_round_1)
        
        # Agents submit bids for the second round
        bids_round_2 = jnp.array([8.0, 5.0, 7.0])
        state, timestep = env.step(state, bids_round_2)
        env.render(state, bids_round_2)
        
    run_auction()