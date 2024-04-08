from flask import Flask, render_template, request, jsonify
from serpapi import GoogleSearch
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

@app.route('/')
def home():
    return render_template('index.html')

# Define the directory to save the images
OUTPUT_DIR = 'E:\output image'

# Create the directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        print("Start of upload route")
        
        # Get the uploaded image, category, and style prompt from the request
        image_file = request.files['image']
        image = Image.open(image_file)
        category = request.form['category']
        style_prompt = request.form['stylePrompt']

        print("Image loaded")

        # Remove the resizing step to keep the original size of the uploaded image
        # image = image.convert("RGB").resize((512, 512))
        #image = image.convert("RGB")

        # Use the original image for manipulation
        #input_image = torch.tensor(np.array(image)).unsqueeze(0).to(pipe.device)


        # Convert the image to a suitable format for the model
        image = image.convert("RGB").resize((512, 512))
        input_image = torch.tensor(np.array(image)).unsqueeze(0).to(pipe.device)

        print("Image converted")

        # Generate the image using the provided style prompt
        generated_image = pipe(prompt=style_prompt, image=input_image, strength=0.65, num_inference_steps=60)

        print("Image generated")

        # Save the generated image with a unique filename
        filename = f'{category}_generated_image.png'
        filepath = os.path.join(OUTPUT_DIR, filename)
        generated_image.images[0].save(filepath, format="PNG")

        # Save the generated image
        output_buffer = BytesIO()
        generated_image.images[0].save(output_buffer, format="PNG")
        output_buffer.seek(0)

        print("Image saved")

        # Return the saved image path
        base64_encoded_image = base64.b64encode(output_buffer.read()).decode()

        # Return the base64-encoded image in the response
        return jsonify({"imageUrl": "data:image/png;base64," + base64_encoded_image})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)})
    
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return "Please set the SERP_API_KEY environment variable."
    
    params = {
        "api_key": api_key,
        "engine": "google_shopping",
        "google_domain": "google.com",
        "q": query
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    app.logger.info('Search results retrieved successfully:', results)
    
    if 'shopping_results' not in results:
        app.logger.warning('No shopping results found')
        return jsonify([])  # Return an empty list if no results found
    
    # Extract relevant information from results
    formatted_results = []
    for item in results['shopping_results']:
        formatted_result = {
            'title': item['title'],
            'thumbnail': item['thumbnail'],
            'price': item['price'],
            'source': item['source'],
            'link': item['link']
        }
        formatted_results.append(formatted_result)
    
    return jsonify(formatted_results)

if __name__ == '__main__':
    app.run(port=5000, debug=False)
