from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import os
from www.src.utils_gpt_based import create_resut_file, create_topics_table
from www.src.utils import decompress_and_read_files, svm_based, bert_based
import asyncio
import nest_asyncio
import joblib
import shutil
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
clf = joblib.load("trained_svc_model.pkl")

bert_model = AutoModelForSequenceClassification.from_pretrained("bert_model", num_labels=40, problem_type="multi_label_classification")
tokenizer = AutoTokenizer.from_pretrained("bert_model")


nest_asyncio.apply()

css_style = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f6f9;
        color: #333;
        margin: 0;
        padding: 0;
    }
    h3 {
        font-size: 24px;
        font-weight: 600;
        text-align: center;
        color: #1f78b4;
        margin-bottom: 20px;
    }
    .panel-sidebar {
        background-color: #ffffff;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for a clean look */
    }
    .panel-main {
        background-color: #ffffff;
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for a clean look */
    }
    .shiny-input-container {
        margin-bottom: 15px;
    }
    .shiny-download-button, .shiny-action-button {
        background-color: #1f78b4;
        color: white;
        border: none;
        padding: 10px 15px;
        font-size: 14px;
        text-transform: uppercase;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s;
        border-radius: 0;
        width: 100%;
    }
    .shiny-download-button:hover, .shiny-action-button:hover {
        background-color: #145a86;
    }
    .shiny-input-container label {
        font-weight: 600;
        color: #1f78b4;
    }
    .navset-tab .nav {
        background-color: #1f78b4;
        color: white;
        padding: 10px 15px;
        margin-right: 5px;
        cursor: pointer;
        text-transform: uppercase;
        font-size: 14px;
    }
    .navset-tab .nav.active {
        background-color: #145a86;
    }
</style>
"""

app_ui = ui.page_fluid(
    ui.tags.style(css_style),
    ui.tags.script(
        """
        Shiny.addCustomMessageHandler('disable_button', function(buttonId) {
            $('#' + buttonId).prop('disabled', true);
        });
        
        Shiny.addCustomMessageHandler('enable_button', function(buttonId) {
            $('#' + buttonId).prop('disabled', false);
        });

        Shiny.addCustomMessageHandler('disable_link', function(linkId) {
            $('#' + linkId).css({
                'pointer-events': 'none',
                'color': 'gray',
                'text-decoration': 'none',  // Remove underline for a clearer disabled look
                'background-color': 'white'
            });
        });

        Shiny.addCustomMessageHandler('enable_link', function(linkId) {
            $('#' + linkId).css({
                'pointer-events': 'auto',
                'color': '',
                'text-decoration': '',  // Restore original text decoration
                'background-color': ''
            });
        });

        """
    ),
    ui.h3("Qualitative Data Analysis", style="text-align: center; color: #333; margin-bottom: 20px;"),
    ui.navset_tab(
        ui.nav_panel("Load Data",
            ui.panel_sidebar(
                ui.input_file("archive", "Choose an archive file with qualitative data", accept=[".zip"]),
                width=2.5,
                class_="panel-sidebar"
            )
        ),
        ui.nav_panel("Main",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_selectize("process_type", "Select interview to observe",
                        choices=["GPT_based", "SVM", "Berta classifier"], selected="GPT_based", multiple=False),
                    ui.download_button("download_topics", "Download Topics Table", class_="shiny-download-button"),
                    ui.download_button("download_matrix", "Download Result Matrix", class_="shiny-download-button"),
                    width=2.5,
                    class_="panel-sidebar"
                )
            )
        ),
    )
)

def server(input, output, session):

    data_raw = reactive.Value(None)
    topics_table = reactive.Value(None)

    if not os.path.exists("output_dir"):
        os.mkdir("output_dir")
    else:
        shutil.rmtree("output_dir")
        os.mkdir("output_dir")

    if not os.path.exists("archive_dir"):
        os.mkdir("archive_dir")
    else:
        shutil.rmtree("archive_dir")
        os.mkdir("archive_dir")
    @reactive.Effect
    @reactive.event(input.archive)
    def load_file():
        print(input.archive()[0]["datapath"])
        if input.archive() is not None:
            archive_path = input.archive()[0]["datapath"]
            df = decompress_and_read_files(archive_path)
            data_raw.set(df)

    @render.download()
    def download_topics():
        if data_raw() is not None:
            process_type = input.process_type()
            # GPT based
            if process_type == "GPT_based":
                res = asyncio.run(create_topics_table(data_raw()))
                topics_table.set(res)
                res.to_excel("output_dir/res.xlsx", index=False)
            
            # SVM based
            elif process_type == "SVM":
                res = svm_based(data_raw(), clf, embedding_model)
                topics_table.set(res)
                res.to_excel("output_dir/res.xlsx", index=False)
                
            # Berta based
            else:
                res = bert_based(data_raw(), bert_model, tokenizer)
                topics_table.set(res)
                res.to_excel("output_dir/res.xlsx", index=False)

            return "output_dir/res.xlsx"
        
    @render.download()
    def download_matrix():
        if topics_table() is not None:
            result_df = create_resut_file(topics_table.get())
            for file in os.listdir("output_dir"):
                os.remove(f"output_dir/{file}")
            result_df.save("output_dir/result_table.xlsx")
            return "output_dir/result_table.xlsx"
        else:
            ui.notification_show("No topics table was loaded", duration=20, type = "error")
            return "empty.xlsx"

app = App(app_ui, server)
app.run()