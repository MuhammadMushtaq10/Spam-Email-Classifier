import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

spam_data = pd.read_csv("spam1.csv" )

#inspect
print(spam_data.groupby('Category').describe())
spam_data['spam'] = spam_data['Category'].apply(lambda x: 1 if x =='spam' else 0)
print(spam_data)

x_train , x_test , y_train , y_test = train_test_split(spam_data.Message , spam_data.spam)

print(x_train.describe())

#find the total word count and store the data in numerical matrix
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

print(x_train_count.toarray())

#train model
model = MultinomialNB()
model.fit(x_train_count,y_train)

print(MultinomialNB())

#testing ham
email_ham = ["Meet me in library ASAP"]
email_ham_count = cv.transform(email_ham)
model.predict(email_ham_count)

print(model.predict(email_ham_count))

#testing spam
email_spam = ["click the link below to win"]
email_spam_count = cv.transform(email_spam)
model.predict(email_spam_count)

print(model.predict(email_spam_count))

#testing model
x_test_count = cv.transform(x_test)
model.score(x_test_count,y_test)

print(model.score(x_test_count,y_test))

# Function to predict if the email is spam or ham
def predict_spam():
    email_text = text_entry.get()  # Get the text from the entry field
    if not email_text:
        messagebox.showinfo("Invalid Input", "Please enter some text")
        return
    
    # Simulate a loading effect
    loading_label.config(text="Detecting...")
    root.update_idletasks()  # Forces an update to show the animation
    time.sleep(1)  # Simulate processing time
    
    # Transform the input text using the CountVectorizer
    email_count = cv.transform([email_text])
    
    # Predict if it's spam or ham
    prediction = model.predict(email_count)
    
    # Display the result in a styled message box
    if prediction[0] == 1:
        result_text = "❌ This email is spam!"
        messagebox.showinfo("Spam Detection Result", result_text, icon="error")
    else:
        result_text = "✅ This email is not spam!"
        messagebox.showinfo("Spam Detection Result", result_text, icon="info")
    
    # Clear the loading text
    loading_label.config(text="")
    
    # Log the prediction in the prediction history
    prediction_history.insert(tk.END, f"Input: {email_text}\nResult: {result_text}\n---\n")

# Function to create a spam word cloud
def create_word_cloud():
    # Sample text data for creating a word cloud (could be generated from actual spam data)
    spam_text = "Congratulations! You've won a free iPhone. Click here to claim. Win a free cruise."
    
    # Generate the word cloud
    wordcloud = WordCloud(width=400, height=400, background_color="white").generate(spam_text)
    
    # Create a figure for the word cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")  # Hide the axes
    
    # Create a new window for the word cloud
    wordcloud_window = tk.Toplevel(root)
    wordcloud_window.title("Spam Word Cloud")
    
    # Display the word cloud on the canvas
    canvas = FigureCanvasTkAgg(fig, wordcloud_window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Function to save the prediction history to a file
def save_history():
    with open("prediction_history.txt", "w") as f:
        f.write(prediction_history.get("1.0", tk.END))
    messagebox.showinfo("Save History", "Prediction history saved to prediction_history.txt")

# Function to load prediction history from a file
def load_history():
    try:
        with open("prediction_history.txt", "r") as f:
            history_text = f.read()
        prediction_history.delete("1.0", tk.END)
        prediction_history.insert(tk.END, history_text)
        messagebox.showinfo("Load History", "Prediction history loaded.")
    except FileNotFoundError:
        messagebox.showinfo("Load History", "No saved history found.")

# Create a Tkinter window with enhanced visual elements
root = tk.Tk()
root.title("Spam Detection with Naive Bayes")
root.geometry("600x500")  # Larger window for additional elements
root.configure(bg="#303F9F")  # Background color for the window (blue-themed)

# Create a notebook (tabs) to organize the layout
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill=tk.BOTH)

# Create a frame for spam prediction functionality
spam_frame = ttk.Frame(notebook, padding=10)
notebook.add(spam_frame, text="Spam Prediction")

# Create a frame for data visualization (chart)
chart_frame = ttk.Frame(notebook, padding=10)
notebook.add(chart_frame, text="Data Visualization")

# Create a frame for additional features
features_frame = ttk.Frame(notebook, padding=10)
notebook.add(features_frame, text="Additional Features")

# Create the spam prediction widgets
label = ttk.Label(spam_frame, text="Enter email text:", foreground="#00008B")
label.pack(pady=(10, 5))  # Padding to the top and bottom

# Tooltip text when hovering over the label
def show_tooltip(event):
    tooltip.config(text="Enter the text to check for spam.")

def hide_tooltip(event):
    tooltip.config(text="")

label.bind("<Enter>", show_tooltip)  # Hover event for tooltips
label.bind("<Leave>", hide_tooltip)

# Create a tooltip label
tooltip = ttk.Label(spam_frame, text="", foreground="#FFC107", font=("Helvetica", 10))
tooltip.pack()

# Create a text entry field for spam prediction
text_entry = ttk.Entry(spam_frame, width=105)
text_entry.pack(pady=(0, 105))

# Create a loading label
loading_label = ttk.Label(spam_frame, text="", foreground="#FFEB3B", font=("Helvetica", 12, "bold"))
loading_label.pack(pady=(0, 105))

# Create a button for spam detection
predict_button = ttk.Button(spam_frame, text="🔍 Check if it's Spam", command=predict_spam)
predict_button.pack(pady=(0, 20))

# Create a text area for prediction history
prediction_history = ScrolledText(spam_frame, width=50, height=6, wrap=tk.WORD)
prediction_history.pack(pady=(0, 15))

# Create a pie chart for data visualization
fig, ax = plt.subplots()
ax.pie([80, 20], labels=["Non-Spam", "Spam"], autopct="%1.1f%%", startangle=90)
ax.axis("equal")

# Add the pie chart to the chart frame
canvas = FigureCanvasTkAgg(fig, chart_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Add additional features to the additional features frame
create_word_cloud_button = ttk.Button(features_frame, text="Generate Spam Word Cloud", command=create_word_cloud)
create_word_cloud_button.pack(pady=(10, 10))  # Padding for spacing

save_history_button = ttk.Button(features_frame, text="Save Prediction History", command=save_history)
save_history_button.pack(pady=(10, 10))

load_history_button = ttk.Button(features_frame, text="Load Prediction History", command=load_history)
load_history_button.pack(pady=(10, 10))

# Run the Tkinter event loop
root.mainloop()
