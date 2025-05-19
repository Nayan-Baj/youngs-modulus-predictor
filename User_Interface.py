import customtkinter
import pandas as pd
import joblib

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("500x350")

frame = customtkinter.CTkFrame(master = root)
frame.pack(pady = 20, padx = 60, fill = "both", expand = True)

result_label = customtkinter.CTkLabel(master=frame, text="", font=("Roboto", 16))
result_label.pack(pady=(10, 0))

label = customtkinter.CTkLabel(master = frame,
                               text = "Young's Modulus Predictor",
                               font=("Roboto", 24))
label.pack(pady=12, padx=10)

feature_sets = {
    "Density, Debye, Band Gap, Mass, Radius": ['density', 'debye_temperature', 'band_gap', 'avg_atomic_mass', 'avg_atomic_radius'],
    "Density, Debye, Mass, Radius": ['density', 'debye_temperature', 'avg_atomic_mass', 'avg_atomic_radius'],
    "Debye Temp, Volume, Mass": ['debye_temperature', 'volume', 'avg_atomic_mass'],
    "Debye Temp, Mass, Electronegativity": ['debye_temperature', 'avg_atomic_mass', 'avg_electronegativity']
}
model_files = {
    "Density, Debye, Band Gap, Mass, Radius": "model_1.pkl",
    "Density, Debye, Mass, Radius": "model_2.pkl",
    "Debye Temp, Volume, Mass": "model_3.pkl",
    "Debye Temp, Mass, Electronegativity": "model_4.pkl"
}

selected_option = customtkinter.StringVar(value = list(feature_sets.keys())[0])

entry_widgets = {}

def on_predict():
    inputs = {}
    for feature, (label, entry) in entry_widgets.items():
        try:
            value = float(entry.get())
            inputs[feature] = value
        except ValueError:
            print(f"Invalid input for {feature}")
            return  # stop if any input is invalid
    df = pd.DataFrame([inputs])
    selected_label = selected_option.get()
    model_file = model_files.get(selected_label)
    if not model_file:
        print("No model mapped for this feature set")
        return
    try:
        model = joblib.load(model_file)
        prediction = model.predict(df)[0]
        result_label.configure(text=f"Predicted E: {prediction:.2f} GPa")
    except Exception as e:
        result_label.configure(text="Prediction failed")
        print("Prediction Failed", e)
predict_button_packed = False
predict_button = customtkinter.CTkButton(master=frame, text="Predict", command=on_predict)
predict_button.pack(pady=20)

def update_input_fields(choice):
    global predict_button_packed
    #Removes old widgets
    for widget in entry_widgets.values():
        widget[0].destroy()
        widget[1].destroy()
    entry_widgets.clear()

    #Create new widgets for selected features
    features = feature_sets[choice]
    for i, feature in enumerate(features):
        label = customtkinter.CTkLabel(master = frame,
                                       text = feature,
                                       font = ("Roboto", 12))
        label.pack(pady=(5, 0))
        entry = customtkinter.CTkEntry(master = frame)
        entry.pack(pady=(0, 10))
        entry_widgets[feature] = (label, entry)

dropdown = customtkinter.CTkOptionMenu(
    master = frame,
    values = list(feature_sets.keys()),
    variable = selected_option,
    command = update_input_fields
)

dropdown.pack(pady = 10)

update_input_fields(selected_option.get())
root.mainloop()