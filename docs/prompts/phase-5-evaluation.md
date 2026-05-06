---
date: 2026-05-06
phase: 5
topic: "Merge PR #4 + Phase 5 evaluation on vast.ai"
source: user via Claude chat
---

Phase 4 PR #4 (https://github.com/chjs/cacheblend-hf/pull/4)을 머지하고 Phase 5 (Evaluation)를 진행하세요. Phase 5는 vast.ai를 본격 사용하는 첫 단계입니다.
사전 — 프롬프트 아카이브
본 프롬프트를 docs/prompts/phase-5-evaluation.md에 frontmatter(date, phase=5, topic="Merge PR #4 + Phase 5 evaluation on vast.ai") + 본문 그대로 저장. 별도 commit "Archive prompt: phase-5-evaluation"로 push.
Step 1 — PR #4 머지
gh pr checks 4 --repo chjs/cacheblend-hf 로 CI 확인. 실패면 stop & 보고.
gh pr merge 4 --repo chjs/cacheblend-hf --squash --delete-branch
git checkout main && git pull && git branch -d phase-4-pipelining
git checkout -b phase-5-evaluation
Step 2 — Local prep (vast.ai 띄우기 전)
GPU 없이 미리 만들 수 있는 인프라 먼저.
2.1 데이터셋 로더
benchmarks/datasets/musique.py (최우선):
load(split="validation", limit=None) -> list[dict] 구조. 각 item은 {'id', 'system', 'documents', 'query', 'answer'}. HuggingFace datasets 라이브러리로 dgslibisey/MuSiQue 또는 동급 미러 사용. 게이트되어 있으면 datasets.load_dataset 시 trust_remote_code 등 처리. documents는 retriever output(top-k passages) 그대로 list. Phase 5에서는 dataset이 제공한 supporting paragraphs 사용 (별도 retriever 안 띄움 — 본 phase 본질은 알고리즘 비교).
benchmarks/datasets/twiki.py (선택, Musique 통과 후): 2WikiMultihopQA. 같은 인터페이스.
SAMSum/MultiNews는 Phase 5 본 작업 후 여유 시 도입.
2.2 Metrics
benchmarks/metrics/qa.py: f1_score(pred, gold) -> float. token-level F1, 논문 컨벤션 (unicode normalize, lowercase, articles 제거 등 표준 squad-style). aggregate(scores) -> dict. mean, std, p50, p95.
benchmarks/metrics/summarization.py (옵션, SAMSum/MultiNews 도입 시): rouge_l(pred, gold) -> float. rouge_score 라이브러리 사용.
2.3 Benchmark runner
benchmarks/run_benchmark.py: CLI는 --model, --dataset, --method, --ratio, --limit, --storage, --output. --method는 full_recompute, prefix_cache, full_reuse, cacheblend 중 하나. 각 sample은 build input → run method → greedy decode (max_new_tokens=20 for QA) → compute F1. TTFT는 benchmarks/ttft.py 사용. GPU에서는 cuda.synchronize 분기. 결과 JSON 출력은 method, dataset, model, n, f1_mean, f1_std, ttft_median_ms, ttft_p95_ms, kv_hit_rate.
2.4 RAG input builder
benchmarks/rag.py: build_rag_input(item, tokenizer, chunk_size=512) -> dict. dataset item을 system, documents (list of chunks), query 형태로 변환. system은 짧은 instruction prompt, documents는 supporting paragraphs를 chunk_size로 분할. Tokenizer 경계 mismatch 측정 함수 count_tokenizer_mismatches(items, tokenizer) -> dict도 같이.
2.5 Local sanity (GPU 없이)
Qwen2.5-1.5B-Instruct + Musique 5 sample + 4 method 모두 돌아가는지 확인. CPU FP32, S ≤ 1500 정도로 제한. 각 method가 logit/answer를 생산하고 F1이 0~1 사이의 합리적 값을 내는지. 이 단계는 갑작스러운 OOM/dimension mismatch를 vast.ai 전에 잡는 보험. 1-2시간 걸려도 OK.
Step 3 — vast.ai로 본 측정
3.1 인스턴스 띄우기
bash scripts/vast.sh search 로 적절한 offer (GPU ≥24GB, NVMe, verified). RTX 4090 또는 A40 24GB 권장. bash scripts/vast.sh up <offer_id>로 .vast_id 자동 저장. 30초 대기 후 bash scripts/vast.sh ssh로 booting 확인 후 exit.
3.2 코드 sync + 환경 셋업
bash scripts/vast.sh push (rsync). vast.ai SSH로 들어가서: cd /workspace/cacheblend-hf && pip install -r requirements.txt && pip install -e .  실행. python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())" 로 GPU 확인. HF_HOME=/workspace/hf_cache 환경 변수 (vast.sh의 onstart-cmd가 /etc/environment에 등록한 것이 새 SSH 세션에서 적용되는지 확인. 안 되면 export 직접). huggingface-cli login (HF_TOKEN 사용) — Mistral-7B는 게이트되어 있을 수 있음. 모델 한 번 다운로드: python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2'); AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', torch_dtype='float16')" — /workspace/hf_cache에 저장됨.
3.3 Phase 1 회귀 (Mistral-7B로)
Phase 1에서 deferred했던 Mistral-7B layerwise bit-exact 검증을 먼저: pytest tests/test_layerwise.py::test_layerwise_matches_standard -v -s -k mistral (parametrize에 Mistral-7B 추가 필요. FP16 GPU에서는 max_diff < 1e-3 tolerance 적절). 통과 후 결과 수치를 Phase 5 보고서의 "Phase 1 회귀" 섹션에 기록.
3.4 본 측정 — 4-way 비교
Mistral-7B-Instruct-v0.2, FP16 GPU, Musique 50 sample, chunk_size=512, top-k=6:
for method in full_recompute prefix_cache full_reuse cacheblend; do python -m benchmarks.run_benchmark --model mistralai/Mistral-7B-Instruct-v0.2 --dataset musique --limit 50 --method method --ratio 0.15 --storage nvme --output benchmarks/results/musique_{method}.json; done
결과 표는 method, F1 mean, F1 std, TTFT median (ms), TTFT p95 (ms) 컬럼.
3.5 Sensitivity (시간 허락 시)
--ratio sweep on 30 samples for {0.05, 0.10, 0.15, 0.20, 0.30}. chunk_size sweep for {256, 512, 1024} (논문 Fig 15 부분 재현).
3.6 Insight 2 재측정 (RAG 데이터)
Phase 4에서 합성 입력의 overlap이 0.26-0.50으로 떨어졌던 것을 Musique input으로 재측정. Spearman rank correlation도 측정 (paper와 동일 metric). 5-10 sample에서 layer 1 vs layer L-1 deviation rank correlation. Phase 4 보고서의 합성 결과와 나란히 비교.
3.7 LoadingController 실측
LoadingController.profile() Mistral-7B FP16 GPU에서 실행 → prefill_per_token_s 실측. NVMe throughput 실측 (dd 또는 python으로 KV size 단위 read benchmark). pick_recompute_ratio() 추천 결과 표 (RAM/NVMe/slow disk). Phase 4의 Mac CPU 결과와 비교.
Step 4 — 결과 회수 + 인스턴스 stop
bash scripts/vast.sh pull (results, reports rsync down). bash scripts/vast.sh stop (GPU 비용 멈춤, 디스크 hf_cache 보존).
제약
Phase 5 budget: 첫 실측은 Musique 50 sample만 통과해도 충분. 다른 데이터셋, sensitivity sweep은 시간/비용 여유 시. vast.ai 비용 의식: 매 작업 후 stop, 한 세션 4-6시간 이내 권장. 새 의존성 (datasets, rouge_score)는 requirements.txt에 추가. 3번 시도 후 막히면 stop & 보고.
마무리
모든 측정 완료 후: pytest -v -m "requires_model and not slow" (전 phase 회귀) Mac CPU에서 통과 확인. pytest -v -m "not gpu and not slow and not requires_model" 통과. python scripts/verify_phase.py --phase 5 통과.
reports/phase-5-report.md 작성. 명시 항목: Mistral-7B layerwise 회귀 (max_diff). Musique 4-way 결과 표 (F1, TTFT). F1 손실 cacheblend(15%) vs full_recompute의 차이 — 목표 ≤ 0.02. TTFT 단축 cacheblend vs full_recompute 비율 — 목표 ≥ 1.5×. Insight 2 RAG 재측정 결과 (overlap + Spearman). LoadingController 실측 결과. Tokenizer mismatch 비율 (RAG input에서). vast.ai 사용 비용/시간 합계. Phase 5 한계 및 향후 작업 (논문 대비 못 미친 부분, 추가 데이터셋, gradual narrowing 도입 여부 결정). GitHub PR URL. "Prompt archive: docs/prompts/phase-5-evaluation.md" cross-ref.
python scripts/update_status.py --phase 5 --status completed. git commit -m "Phase 5: evaluation on Musique with Mistral-7B" && git push -u origin phase-5-evaluation. gh pr create --repo chjs/cacheblend-hf --base main --head phase-5-evaluation --title "Phase 5: evaluation" --body-file reports/phase-5-report.md. python scripts/send_report.py --phase 5.
시작 전 30초 stop-and-think: GOAL.md 재확인. "Phase 5의 본질은 paper-grade 비교가 아니라 우리 구현이 실제 RAG 데이터에서 작동함을 보이는 것. F1 손실 ≤ 0.02, TTFT 단축 ≥ 1.5× 가 1차 목표."
