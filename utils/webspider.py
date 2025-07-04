from selenium import webdriver
import os
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up the web driver (make sure to download the appropriate driver for your browser)
driver = None

def scrape_images(query, num_images, save_path):
    driver = webdriver.Chrome()
    # Create a Google Images search URL
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"

    # Open the Google Images search page
    driver.get(search_url)

    # Scroll down to load more images
    for _ in range(num_images * 5):
        driver.execute_script("window.scrollBy(0,10000)")

    # Wait for the images to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "g-img.mNsIhb")))

    # Get image elements
    img_elements = driver.find_elements(By.CSS_SELECTOR, "g-img.mNsIhb")

    # Create the save directory on the desktop
    project_path = os.path.dirname(os.path.abspath(__file__))
    desktop_path = os.path.join(project_path, "data")
    save_path = os.path.join(desktop_path, save_path)
    os.makedirs(save_path, exist_ok=True)

    # Loop through the first num_images images
    for i, img_element in enumerate(img_elements[:num_images]):
        try:
            # Click on each image to open it
            img_element.click()

            # Wait for the opened image to load
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'img.sFlh5c.FyHeAf')))

            # Get the URL of the opened image
            img_url_element = driver.find_element(By.CSS_SELECTOR, 'img.sFlh5c.FyHeAf')
            img_url = img_url_element.get_attribute("src")

            # Download the image
            img_name = f"{query}_{i+1}.jpg"
            img_path = os.path.join(save_path, img_name)
            response = requests.get(img_url, stream=True)
            with open(img_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"Image {i+1} downloaded successfully")

        except Exception as e:
            print(f"Failed to download image {i+1}: {e}")

def find_cats(num_images):

    save_path = "./downloaded_images"
    cat_name = ['Ragdolls', 'Singapura_cats', 'Persian_cats', 'Sphynx_cats', 'Pallas_cats']
    for cat in cat_name:
        scrape_images(cat, num_images, save_path)

    # Close the browser
    driver.quit()

if __name__ == '__main__':
    query = "dogs"
    num_images = 1000
    find_cats(query, num_images)