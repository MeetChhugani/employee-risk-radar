import json

def main():
    notebook_path = "employee_attrition_predictor.ipynb"
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    for i in [9, 10]:
        cell = nb["cells"][i]
        print(f"\n--- Cell {i} Source ---")
        print("".join(cell["source"]))
        print(f"\n--- Cell {i} Outputs ---")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                print("".join(out["text"]))

if __name__ == "__main__":
    main()
