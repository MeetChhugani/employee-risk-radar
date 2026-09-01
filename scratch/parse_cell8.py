import json

def main():
    notebook_path = "employee_attrition_predictor.ipynb"
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cell = nb["cells"][8]
    print("--- Cell 8 Source ---")
    print("".join(cell["source"]))
    print("\n--- Cell 8 Outputs ---")
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            print("".join(out["text"]))

if __name__ == "__main__":
    main()
