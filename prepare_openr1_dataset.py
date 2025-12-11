from datasets import load_dataset

def main():
    print("Loading OpenR1 dataset...")
    raw = load_dataset("open-r1/OpenR1-Math-220k", "default")

    def convert(example):
        problem = example["problem"]
        solution = example["solution"]

        prompt = f"""
You are a strong reasoning model. Solve the following math problem step by step.

Problem:
{problem}
"""
        return {
            "instruction": prompt.strip(),
            "input": "",
            "output": solution.strip()  
        }

    print("Converting format...")
    converted = raw.map(convert)
    converted = converted["train"]

    print("Saving JSONL...")
    converted.to_json(
        "openr1_math_220k_distill.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print("Dataset saved to: openr1_math_220k_distill.jsonl")

if __name__ == "__main__":
    main()

''