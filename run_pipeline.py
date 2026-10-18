"""
Runs the full build in order. After it finishes, start the dashboard with:
    streamlit run app.py

Steps:
  1-3  train the replay model, simulate logs, score and explain
  4    train the live attack-type classifier (Feature 1)
  5    generate the result figures (ROC, PR, model comparison) for the report
  6    measure the security/explainability overhead
"""
import subprocess
import sys

scripts = [
    "step1_train_model.py",
    "step2_simulate_logs.py",
    "step3_score_and_explain.py",
    "live_classifier.py",        # Feature 1: live attack-type classifier
    "make_result_figures.py",    # ROC / PR / comparison figures
    "benchmark_overhead.py",     # security-component overhead
]

for script in scripts:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, script], check=True)

print("\nAll done. Now run:  streamlit run app.py")
