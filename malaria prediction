import tkinter as tk
from tkinter import ttk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

class MalariaAnalysisApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Malaria Data Analysis and Prediction")
        self.master.geometry("1200x700")

        # Load and preprocess the data
        self.data = self.load_data()

        # GUI Elements
        self.create_widgets()

    def load_data(self):
        # Load malaria dataset
        data = pd.read_excel(r"C:\Users\Administrator\Desktop\class work\malaria.xlsx")


        # Convert 'date' column to datetime and sort
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(by='date')
        data.set_index('date', inplace=True)

        return data

    def create_widgets(self):
        # Region selection
        ttk.Label(self.master, text="Select Region:").grid(row=0, column=0, padx=10, pady=10)
        self.region_var = tk.StringVar()
        self.region_combobox = ttk.Combobox(self.master, textvariable=self.region_var)
        self.region_combobox['values'] = ['All Regions'] + list(self.data['region'].unique())
        self.region_combobox.grid(row=0, column=1)
        self.region_combobox.bind("<<ComboboxSelected>>", self.update_counties)

        # County selection
        ttk.Label(self.master, text="Select County:").grid(row=0, column=2, padx=10, pady=10)
        self.county_var = tk.StringVar()
        self.county_combobox = ttk.Combobox(self.master, textvariable=self.county_var)
        self.county_combobox.grid(row=0, column=3)

        # Analysis type
        ttk.Label(self.master, text="Analysis Type:").grid(row=1, column=0, padx=10, pady=10)
        self.analysis_var = tk.StringVar()
        self.analysis_combobox = ttk.Combobox(self.master, textvariable=self.analysis_var)
        self.analysis_combobox['values'] = ["Total Cases", "Severe Cases", "Deaths", "Mosquito Density"]
        self.analysis_combobox.current(0)
        self.analysis_combobox.grid(row=1, column=1)

        # ARIMA inputs
        ttk.Label(self.master, text="Weeks to Predict:").grid(row=2, column=0)
        self.weeks_var = tk.IntVar(value=4)
        ttk.Entry(self.master, textvariable=self.weeks_var).grid(row=2, column=1)

        ttk.Label(self.master, text="ARIMA p:").grid(row=2, column=2)
        self.p_var = tk.IntVar(value=1)
        ttk.Entry(self.master, textvariable=self.p_var, width=5).grid(row=2, column=3)

        ttk.Label(self.master, text="d:").grid(row=2, column=4)
        self.d_var = tk.IntVar(value=1)
        ttk.Entry(self.master, textvariable=self.d_var, width=5).grid(row=2, column=5)

        ttk.Label(self.master, text="q:").grid(row=2, column=6)
        self.q_var = tk.IntVar(value=1)
        ttk.Entry(self.master, textvariable=self.q_var, width=5).grid(row=2, column=7)

        # Buttons
        ttk.Button(self.master, text="Update Analysis", command=self.update_analysis).grid(row=3, column=0, pady=10)
        ttk.Button(self.master, text="Run Prediction", command=self.run_prediction).grid(row=3, column=1, pady=10)

        # Text output area
        self.analysis_text = tk.Text(self.master, height=15, width=80)
        self.analysis_text.grid(row=4, column=0, columnspan=8, padx=10, pady=10)

        # Matplotlib Figure
        self.figure = plt.Figure(figsize=(10, 5), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.master)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=8)

    def update_counties(self, event=None):
        region = self.region_var.get()
        if region == 'All Regions':
            counties = sorted(self.data['county'].unique())
        else:
            counties = sorted(self.data[self.data['region'] == region]['county'].unique())
        self.county_combobox['values'] = counties
        self.county_combobox.set(counties[0] if counties else '')

    def update_analysis(self):
        region = self.region_var.get()
        county = self.county_var.get()
        analysis_type = self.analysis_var.get()

        filtered_data = self.data
        if region != 'All Regions':
            filtered_data = filtered_data[filtered_data['region'] == region]
        if county:
            filtered_data = filtered_data[filtered_data['county'] == county]

        if analysis_type == "Total Cases":
            data_series = filtered_data['total_cases']
        elif analysis_type == "Severe Cases":
            data_series = filtered_data['severe_cases']
        elif analysis_type == "Deaths":
            data_series = filtered_data['deaths']
        else:
            data_series = filtered_data['mosquito_density']

        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, f"Analysis for {analysis_type}:\n")
        self.analysis_text.insert(tk.END, f"Region: {region}\nCounty: {county}\n\n")
        self.analysis_text.insert(tk.END, f"Data Summary:\n{data_series.describe()}\n")

        self.plot.clear()
        self.plot.plot(filtered_data.index, data_series, label=analysis_type)
        self.plot.set_title(f"{analysis_type} Over Time")
        self.plot.set_xlabel("Date")
        self.plot.set_ylabel(analysis_type)
        self.plot.legend()
        self.canvas.draw()

    def run_prediction(self):
        region = self.region_var.get()
        county = self.county_var.get()
        weeks_to_predict = self.weeks_var.get()
        p = self.p_var.get()
        d = self.d_var.get()
        q = self.q_var.get()

        filtered_data = self.data
        if region != 'All Regions':
            filtered_data = filtered_data[filtered_data['region'] == region]
        if county:
            filtered_data = filtered_data[filtered_data['county'] == county]

        data_series = filtered_data['total_cases']

        model = ARIMA(data_series, order=(p, d, q))
        model_fit = model.fit()

        forecast_index = [filtered_data.index[-1] + pd.DateOffset(weeks=i) for i in range(1, weeks_to_predict + 1)]
        forecast = model_fit.forecast(steps=weeks_to_predict)

        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, f"Prediction for {weeks_to_predict} weeks:\n")
        for i, pred in enumerate(forecast):
            self.analysis_text.insert(tk.END, f"Week {i+1}: {pred:.2f} predicted total cases\n")

        self.plot.clear()
        self.plot.plot(filtered_data.index, data_series, label="Historical Data")
        self.plot.plot(forecast_index, forecast, label="Predicted Data", linestyle='--')
        self.plot.set_title(f"Total Cases Prediction for {weeks_to_predict} Weeks")
        self.plot.set_xlabel("Date")
        self.plot.set_ylabel("Total Cases")
        self.plot.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MalariaAnalysisApp(root)
    root.mainloop()
