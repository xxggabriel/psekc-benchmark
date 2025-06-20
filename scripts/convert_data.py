import pickle
import os

DATA_FILENAME = '../data/dirna.data'
OUTPUT_CSV_FILENAME = '../data/dirna_properties.csv'

if not os.path.exists(DATA_FILENAME):
    print(f"Erro: Arquivo '{DATA_FILENAME}' não encontrado. Baixe-o primeiro.")
else:
    print(f"Lendo o arquivo pickle '{DATA_FILENAME}'...")
    with open(DATA_FILENAME, "rb") as f:
        data = pickle.load(f)

    first_key = next(iter(data))
    prop_names = [prop_name for prop_name, _ in data[first_key]]

    print(f"Convertendo para '{OUTPUT_CSV_FILENAME}'...")
    with open(OUTPUT_CSV_FILENAME, "w") as f:

        f.write(f"k_tuple,{','.join(prop_names)}\n")
        
        for k_tuple, prop_list in sorted(data.items()):
            values = [str(value) for _, value in prop_list]
            f.write(f"{k_tuple},{','.join(values)}\n")

    print("Conversão concluída com sucesso.")