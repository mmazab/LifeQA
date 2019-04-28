# Changes in the dataset over time

This file traces how the data was moved around to understand the files around.

The questions were originally written in a Google Sheet. Then, they were downloaded into
[questions/combined_question.csv](questions/combined_question.csv). A JSON file was generated (which is now located in
[auto_captioned_data/lqa_data.json](auto_captioned_data/lqa_data.json)) from it, and divided into train, dev and test.
Automatic captions were added to them. Then, these files were moved to the folder
[auto_captioned_data/](auto_captioned_data), and new ones were created which contain both automatic and manual captions
if available, otherwise only automatic data.

So, if any change is done to the dataset (e.g., fixing some question or answer), it should be done for the current
file versions (directly under this directory) or else to all these files mentioned.
