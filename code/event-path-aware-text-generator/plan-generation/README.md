## Event Transition Planning --- PlanGeneration
This folder implements the event transition planner which completes the partial event path given certain input context.

1. we first fine-tune it with large-scaled event transition paths sampled from ATOMIC, transitions from commonsense events. Following are the data statistics:

|      |   train   |   dev   |   test  |
|------|-----------|---------|---------|
|ATOMIC| 3,638,679 | 202,149 | 202,149 |

2. After that, we fine-tune the resulting GPT-2 model in addition on the event transitions extracted from the training corpus, so that the planner is aware of general transitions in the commonsense while focusing on the transitions in the specific domain in the meantime. 
To protect the learned event transition information in the first step. We adopt pre-fix tuning when learning downstream event transition patterns.





