// Composition metadata for the two follow-up studies; each study owns its numerical work.
export const evidenceAuditStudies = [
  { id: "retrieval-comparison", title: "混合召回、候选覆盖与来源组配对评估", pages: ["retrieval-project", "paired-evaluation"], schema: "retrieval-comparison-v1", variable: "retrievalComparison",
    sources: ["scripts/retrieval_comparison.py", "scripts/paired_evidence.py", "scripts/framework-requirements.txt"] },
  { id: "text-evaluation-audit", title: "固定文本模型的逐文档误差与加权评估", pages: ["text-training-project"], schema: "text-evaluation-audit-v1", variable: "textEvaluationAudit",
    sources: ["scripts/text_evaluation_audit.py", "scripts/paired_evidence.py", "scripts/text_training_core.py", "scripts/learning_core.py", "scripts/framework-requirements.txt"] },
];
