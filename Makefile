.PHONY: install install-vllm lint format test data eval-baselines eval-vllm bench compare serve clean

install:
	pip install -r requirements.txt

install-vllm:
	pip install -r requirements-vllm.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest

data:
	python -m data.build_dataset

eval-baselines:
	python -m cli.main eval openai
	python -m cli.main eval gemini

eval-vllm:
	python -m cli.main eval vllm

bench:
	python -m cli.main bench --concurrency 1,4,8,16,32

compare:
	python -m cli.main compare

serve:
	bash service/run.sh

clean:
	rm -rf bench/results/*.json eval/results/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
