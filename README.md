# Auction Environment

Welcome to the **Auction Environment**! This is a reinforcement learning environment simulating a multi-agent auction game, built using [JAX](https://github.com/google/jax) and [Jumanji](https://github.com/instadeepai/jumanji). Agents participate in sequential auctions, bidding against each other to maximize their utilities based on private valuations given by a uniform distribution. Valuations of each agent remain constant for all rounds.

## Usage

### Environment Description

The auction environment simulates the following:

- **Agents**: Multiple agents participate, each with a private valuation for the auctioned item.
- **Valuations**: Agents' valuations are randomly sampled at the beginning and kept private.
- **Bidding**: In each round, agents submit bids simultaneously.
- **Winner Determination**: The agent with the highest bid wins the auction round. In case of a tie, the agent with the lowest index wins.
- **Rewards**: The winning agent receives a reward equal to their valuation minus their bid. Other agents receive zero reward.
- **Cumulative Utility**: Agents accumulate utility over multiple rounds.
- **Observation**: After each round, agents observe the winning agent's index and the winning bid amount.

### Running the Auction

Here's how you can run the auction environment and visualize the auction process:

```python
import jax
import jax.numpy as jnp
from auction import Auction

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
    
if __name__ == "__main__":
    run_auction()
```

### Example Output

```plaintext
=== Auction Round 0 ===
Agent Valuations:
  Agent 0: Valuation = 5.94
  Agent 1: Valuation = 7.44
  Agent 2: Valuation = 6.42

No bids have been submitted yet.
========================

=== Auction Round 1 ===
Agent Valuations:
  Agent 0: Valuation = 5.94
  Agent 1: Valuation = 7.44
  Agent 2: Valuation = 6.42

Agents' Bids:
  Agent 0: Bid = 5.0
  Agent 1: Bid = 7.0
  Agent 2: Bid = 6.0

Winning Agent: Agent 1 with a bid of 7.0

Cumulative Utilities:
  Agent 0: Cumulative Utility = 0.0
  Agent 1: Cumulative Utility = 0.44
  Agent 2: Cumulative Utility = 0.0
========================

=== Auction Round 2 ===
Agent Valuations:
  Agent 0: Valuation = 5.94
  Agent 1: Valuation = 7.44
  Agent 2: Valuation = 6.42

Agents' Bids:
  Agent 0: Bid = 8.0
  Agent 1: Bid = 5.0
  Agent 2: Bid = 7.0

Winning Agent: Agent 0 with a bid of 8.0

Cumulative Utilities:
  Agent 0: Cumulative Utility = -2.06
  Agent 1: Cumulative Utility = 0.44
  Agent 2: Cumulative Utility = 0.0
========================
```