import openai

# Set your OpenAI API key
api_key = "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"

# Initialize the OpenAI API client
openai.api_key = api_key

detection_results = """Label: person, Confidence: 0.946457028388977, Bounding Box: (198, 447, 528, 814) 
Label: person, Confidence: 0.9979164600372314, Bounding Box: (862, 435, 1273, 848) 
Label: person, Confidence: 0.9860677123069763, Bounding Box: (51, 492, 409, 851) 
Label: person, Confidence: 0.9500231742858887, Bounding Box: (531, 336, 596, 516) 
Label: person, Confidence: 0.7680507302284241, Bounding Box: (685, 364, 763, 489) 
Label: person, Confidence: 0.8114284873008728, Bounding Box: (445, 354, 583, 575) 
Label: person, Confidence: 0.9048537015914917, Bounding Box: (737, 395, 849, 579) 
Label: person, Confidence: 0.9979507923126221, Bounding Box: (374, 410, 563, 650) 
Label: person, Confidence: 0.9634515047073364, Bounding Box: (755, 431, 983, 702) 
Label: person, Confidence: 0.564125657081604, Bounding Box: (621, 363, 637, 418) 
Label: handbag, Confidence: 0.8214112520217896, Bounding Box: (858, 528, 950, 599) 
Label: handbag, Confidence: 0.6095970869064331, Bounding Box: (276, 615, 378, 682) 

{"result": {"tags": [{"confidence": "97.74", "type": "People", "tag": "Person", "i18n_tag": {"zh": "人", "en": "Person"}, "i18n_type": {"zh": "人", "en": "People"}, "instances": []}, {"confidence": "91.19", "type": "Natural scenery", "tag": "Sky", "i18n_tag": {"zh": "天空", "en": "Sky"}, "i18n_type": {"zh": "自然风景", "en": "Natural scenery"}, "instances": []}, {"confidence": "85.00", "type": "Manual scenario", "tag": "Carriage", "i18n_tag": {"zh": "车厢", "en": "Carriage"}, "i18n_type": {"zh": "人工场景", "en": "Manual scenario"}, "instances": []}]}}
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "system", "content": """You are a wearable for the blind that helps them to describe/navigate their surroundings. Given below are 2 object/image recognition results given by 2 tools. Can you interpret and infer them to describe the environment/surrounding to the blind user in one paragraph. Keep it layman and succinct.\n\n
"""},
        {"role": "user", "content": detection_results},
    ],
)

content = response.choices[0].message.content
print(content)

