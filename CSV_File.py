from mp_api.client import MPRester
import pandas as pd
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure, Element
from joblib import Parallel, delayed
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("apiKey")
#Connects to elastic part of the API
with MPRester(apiKey) as mpr:
    elasticity_docs = mpr.materials.elasticity.search(
        fields = [
            "material_id",
            "bulk_modulus",
            "shear_modulus",
            "debye_temperature"
        ],
        num_chunks= 15,
        chunk_size = 1000
        )

#Conversion again
elasticity_docs_as_dicts = [doc.model_dump() for doc in elasticity_docs]
elasticity_df = pd.DataFrame(elasticity_docs_as_dicts)
elasticity_df = elasticity_df[["material_id", "bulk_modulus", "shear_modulus", "debye_temperature"]]

material_ids = elasticity_df["material_id"].tolist()
chunk_size = 1000;
chunk_ids = []
summary_docs = []
#Connects to API
with MPRester(apiKey) as mpr:
    #Searching through materials summaries
    for i in range(0, len(material_ids), chunk_size):
        chunk_ids = material_ids[i : i + chunk_size]
        chunk = mpr.materials.summary.search(
            #Pulling anything with elastic values
            material_ids=chunk_ids,
            #Fields we're looking for in this element
            fields = [
                "material_id",
                "formula_pretty",
                "density",
                "band_gap",
                "volume",
                "energy_above_hull",
                "formation_energy_per_atom",
                "structure"
            ],
            num_chunks = 15,
            chunk_size= 1000
            )
        summary_docs.extend(chunk)

#Changing from MPData Objects to python dictionaries
summary_docs_as_dicts = [doc.model_dump() for doc in summary_docs]
summary_df = pd.DataFrame(summary_docs_as_dicts)

#If the item does not have any bulk nor shear moduli
if elasticity_df.empty:
    print("No elasticity data found for these materials")
#If it does
else:
    df = pd.merge(summary_df, elasticity_df, on="material_id", how="inner")
    df = df.rename(columns={
        "bulk_modulus_y": "bulk_modulus",
        "shear_modulus_y": "shear_modulus"
    })
    df = df.dropna(subset=["bulk_modulus", "shear_modulus"])

    def compute_youngs_modulus(row):
        k = row["bulk_modulus"]["vrh"]
        g = row["shear_modulus"]["vrh"]
        return (9 * k * g) / (3 * k + g)

    cnn = CrystalNN()
    def avg_coordination(struct: Structure, nn_finder = cnn) -> float | None:
        coordination_numbers = []
        try:
            for i in range(len(struct)):
                cn = nn_finder.get_cn(struct, i)
                coordination_numbers.append(cn)
        except (ValueError, RuntimeError, IndexError):
            return None
        if len(coordination_numbers) > 0:
            avg_cn = sum(coordination_numbers) / len(coordination_numbers)
            return avg_cn
        else:
            return None

    def get_atomic_features(structure):
        try:
            comp = structure.composition
            total_atoms = sum(comp.values())

            avg_mass = sum(Element(str(el)).atomic_mass * amt for el, amt in comp.items()) / total_atoms
            avg_electronegativity = sum(
                Element(str(el)).X * amt for el, amt in comp.items() if Element(str(el)).X is not None) / total_atoms
            avg_atomic_radius = sum(Element(str(el)).atomic_radius * amt for el, amt in comp.items() if
                                    Element(str(el)).atomic_radius is not None) / total_atoms
            return avg_mass, avg_electronegativity, avg_atomic_radius
        except (ValueError, TypeError, KeyError):
            return None, None, None

    #Youngs Modulus
    df["youngs_modulus"] = df.apply(compute_youngs_modulus, axis=1)
    # Filtering
    df = df[(df["youngs_modulus"] > 1) & (df["youngs_modulus"] < 1000) & (df["energy_above_hull"] <= 0.001)]
    #Coordination Number
    df["avg_coordination_number"] = Parallel(n_jobs = -1, verbose = 5)(
        delayed(avg_coordination)(s) for s in df["structure"]
    )
    atomic_features = df["structure"].apply(get_atomic_features)

    # Atomic_features
    df["avg_atomic_mass"] = atomic_features.apply(lambda x: x[0])
    df["avg_electronegativity"] = atomic_features.apply(lambda x: x[1])
    df["avg_atomic_radius"] = atomic_features.apply(lambda x: x[2])
    df.to_csv("materials_data.csv", index=False)



