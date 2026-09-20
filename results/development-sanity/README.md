# Development diagnostics

Before the test pilot, DeepSeek beta strict label-only output failed validation on 505 of 538 common RAGTruth development cases, usually returning bare yes/no rather than the required JSON object. The GA JSON profile worked and is the DeepSeek configuration used in the published test comparison.

[snapshot.json](snapshot.json) preserves the development counts, predictions, failures, quality and timing summaries for all compared modes. It also contains a partial ANLI diagnostic. These are development observations, not completed test-set results or evidence of generalization.

The older free GLM endpoint returned six overload errors in ten requests; [the diagnostic records](../availability/free-glm-pilot.jsonl) are retained separately. The test pilot uses `glm-5.3-flash`.
