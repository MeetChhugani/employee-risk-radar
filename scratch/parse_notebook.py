import json

def main():
    notebook_path = "employee_attrition_predictor.ipynb"
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        print("Notebook loaded successfully!")
        print("Number of cells:", len(nb["cells"]))
        
        # Look for code outputs containing metrics or accuracy
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if any(x in source.lower() for x in ["auc", "accuracy", "xgboost", "train", "confusion"]):
                    print(f"\n--- Cell {idx} (Code) ---")
                    print(source[:300] + ("..." if len(source) > 300 else ""))
                    # Check outputs
                    for out in cell.get("outputs", []):
                        if out.get("output_type") == "stream":
                            text = "".join(out["text"])
                            print("Output Stream:")
                            print(text[:300] + ("..." if len(text) > 300 else ""))
                        elif out.get("output_type") == "execute_result":
                            data = out.get("data", {}).get("text/plain", "")
                            print("Output Execute Result:")
                            print("".join(data)[:300])
            elif cell["cell_type"] == "markdown":
                source = "".join(cell["source"])
                if "#" in source:
                    print(f"\n--- Cell {idx} (Markdown) ---")
                    print(source)
    except Exception as e:
        print("Failed to read notebook:", e)

if __name__ == "__main__":
    main()
