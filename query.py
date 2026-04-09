# Import libraries
import chromadb
from openai import OpenAI
import csv
from datetime import datetime
import re
import os
from dotenv import load_dotenv
import json
import pandas as pd
from io import StringIO

# Paths

PDF_PATH = "pdf_processing/pdfs"
CLEANED_PDFS_PATH = "pdf_processing/finished_data"


# Get pdfs

pdf_files = [f for f in os.listdir(PDF_PATH) if f.endswith('.pdf')]
paper_ids = [pdf_file.split('-')[0] for pdf_file in pdf_files]


#  environment

load_dotenv() # get your OPENAI_API_KEY from the .env file

pre_prompt = """ 
You are a helpful assistant. You answer questions about diet information in livestock management scientific literature. 
But you only answer based on knowledge I'm providing you. You don't use your internal knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
"""

user_query = """
Return a csv table with information about all the diets in the paper given below. 
The columns should be:
A.Level.Name: The name of the diet, as called in the paper
D.Item: The name of the ingredient in the diet,
D.Type: The ingredient type (Crop Byproduct, Crop Product, Forage Plants, Supplement, and Other Ingredients)
D.Amount: The amount of the ingredient in the diet
D.Unit.Amount: The units for the amount of that ingredient in the diet
DC.Is.Dry: whether the ingredient is dry
D.Ad.lib: whether it was fed ad libitum.
Notes: if available, more information about the ingredient, such as where it was sourced, how it was processed, etc. Wrap this in double quotes.
If information is not given for a specific value, write NA. 
Return only the table in csv format without any extra response.
"""
# Here is an example of a diet csv table: 
# Ingredient,Ingredient Type,Amount,Units,Dry,Fed Ad Libitum
# Yellow maize meal,Crop Product,22.0,%,Yes,Yes
# Eragrostis curvula,Forage Plants,18.0,%,Yes,Yes
# Leucaena leucocephala,Forage Plants,25.0,%,Yes,Yes
# Wheat bran,Crop Byproduct,8.0,%,Yes,Yes

# user_query = input("What do you want to know about this paper?\n") # user query can also come from user input

output_dfs = []
# TODO add error checking, such as if it cannot properly make a df table if the response's csv format is wrong
for PAPER_ID in paper_ids:

    print(f"Currently working on {PAPER_ID}")
    
    # get cleaned pdf text
    json_filename = ""
    for root, dirs, files in os.walk(CLEANED_PDFS_PATH):
        for file in files: 
            if f"cleaned_{PAPER_ID}".lower() in file.lower():
                json_filename = file
                break

    with open(f"{CLEANED_PDFS_PATH}/{json_filename}", 'r') as file:
        cleaned_pdf_dict = json.load(file)

    sections_to_exclude = ["references", "acknowledgements", "conflicts of interest", "conflict of interest declaration", "authors' contributions"]

    cleaned_pdf_text = ""
    for section in cleaned_pdf_dict["sections"]:
        heading = section["heading"]
        if heading and heading.lower() not in sections_to_exclude:
            cleaned_pdf_text += section["heading"] + "\n"
            content = section["content"]
            for line in content:
                cleaned_pdf_text += line + "\n"
            cleaned_pdf_text += "\n"

    # get OpenAI client
    client = OpenAI()

    # define system prompt  
    system_prompt = f"""
    {pre_prompt}
    --------------------
    The data:
    {cleaned_pdf_text}
    """

    # send query to llm
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_query} 
        ]
    )

    # print llm response
    print("\n\n---------------------\n\n")
    response_text = response.choices[0].message.content
    print(response_text)

    # convert string output into dataframe
    # remove first and last lines of ``` from output
    first_newline_ind = response_text.find('\n')
    last_newline_ind = response_text.rfind('\n')
    csv_string = response_text[first_newline_ind+1 : last_newline_ind] 
    # convert to df
    output_df = pd.read_csv(StringIO(csv_string), sep=",")
    # add B.Code column (paper ID) 
    output_df.insert(loc=0, column="B.Code", value=PAPER_ID)

    # add table to dictionary to aggregate them all together at the end
    output_dfs.append(output_df)

all_outputs_df = pd.concat(output_dfs, ignore_index=True)

# print resulting csv to output folder
all_outputs_df.to_csv("all_outputs.csv", index=False)
