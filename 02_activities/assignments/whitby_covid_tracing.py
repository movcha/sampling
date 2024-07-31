import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants representing the parameters of the model
ATTACK_RATE = 0.10
TRACE_SUCCESS = 0.20
SECONDARY_TRACE_THRESHOLD = 2

def simulate_event(m, seed):
    """
    Simulates the infection and tracing process for a series of events.
    
    This function creates a DataFrame representing individuals attending weddings and brunches,
    infects a subset of them based on the ATTACK_RATE, performs primary and secondary contact tracing,
    and calculates the proportions of infections and traced cases that are attributed to weddings.
    
    Parameters:
    - m: Dummy parameter for iteration purposes.
    - seed: Seed for random number generation.
    
    Returns:
    - A tuple containing the proportion of infections and the proportion of traced cases
      that are attributed to weddings.
    """
    np.random.seed(seed + m)
    
    events = ['wedding'] * 200 + ['brunch'] * 800
    ppl = pd.DataFrame({
        'event': events,
        'infected': False,
        'traced': np.nan
    })

    ppl['traced'] = ppl['traced'].astype(pd.BooleanDtype())

    infected_indices = np.random.choice(ppl.index, size=int(len(ppl) * ATTACK_RATE), replace=False)
    ppl.loc[infected_indices, 'infected'] = True

    ppl.loc[ppl['infected'], 'traced'] = np.random.rand(sum(ppl['infected'])) < TRACE_SUCCESS

    event_trace_counts = ppl[ppl['traced'] == True]['event'].value_counts()
    events_traced = event_trace_counts[event_trace_counts >= SECONDARY_TRACE_THRESHOLD].index
    ppl.loc[ppl['event'].isin(events_traced) & ppl['infected'], 'traced'] = True

    ppl['event_type'] = ppl['event'].str[0]
    wedding_infections = sum(ppl['infected'] & (ppl['event_type'] == 'w'))
    brunch_infections = sum(ppl['infected'] & (ppl['event_type'] == 'b'))
    p_wedding_infections = wedding_infections / (wedding_infections + brunch_infections)

    wedding_traces = sum(ppl['infected'] & ppl['traced'] & (ppl['event_type'] == 'w'))
    brunch_traces = sum(ppl['infected'] & ppl['traced'] & (ppl['event_type'] == 'b'))
    p_wedding_traces = wedding_traces / (wedding_traces + brunch_traces)

    return p_wedding_infections, p_wedding_traces

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)

results = [simulate_event(m, seed) for m in range(1000)]
props_df = pd.DataFrame(results, columns=["Infections", "Traces"])

plt.figure(figsize=(10, 6))
sns.histplot(props_df['Infections'], color="blue", alpha=0.75, binwidth=0.05, kde=False, label='Infections from Weddings')
sns.histplot(props_df['Traces'], color="red", alpha=0.75, binwidth=0.05, kde=False, label='Traced to Weddings')
plt.xlabel("Proportion of cases")
plt.ylabel("Frequency")
plt.title("Impact of Contact Tracing on Perceived Infection Sources")
plt.legend()
plt.tight_layout()
plt.show()
