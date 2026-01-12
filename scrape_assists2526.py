from bs4 import BeautifulSoup
import requests
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Step 1: Scrape the initial page for links
def scrape_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    matches = soup.find_all(class_="jsScoreDiv") # Replace 'element' with the actual CSS selector or tag that contains matches
    #let matches = document.getElementsByClassName("jsScoreDiv ")
    links = []
    for match in matches:
        a_tag = match.find('a')
        if a_tag and 'v' not in a_tag.text:
            href = a_tag.get('href')
            links.append(href)

    return links

# Step 2: Use Selenium to execute JavaScript on each link page
def extract_goal_info(urls):
    goal_data = []

    # Set up Selenium WebDriver
    options = Options()
    #options.add_argument('--headless')  # Run headless Chrome
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    for url in urls:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "jsTblVerticalTimeLine"))
        )

        # Execute the JavaScript
        goal_info = driver.execute_script("""
            let goalInfo = {"Home": [], "Away": [], "homeName":document.getElementsByClassName("jsMatchHomeTeam")[0].children[0].children[1].children[0].children[0].children[0].innerHTML, "awayName":document.getElementsByClassName("jsMatchAwayTeam")[0].children[0].children[1].children[0].children[0].children[0].innerHTML, "matchDay":document.getElementsByClassName("jsmatchday")[0].children[0].innerHTML}
            let tables = document.getElementsByClassName("jsTblVerticalTimeLine");
            for (let i = 0; i < tables.length; i++) {
                let events = tables[i].children[0].children;
                for (let j = 0; j < events.length; j++) {
                    let event = events[j].children;
                    for (let k = 0; k < event.length; k++) {
                        let eventInfo = event[k].children[0];
                        if (eventInfo && eventInfo.getAttribute("title") === "Goal") {
                            let scorerInfo = "";
                            let team = "";
                            if (k === 3) {
                                scorerInfo = event[4].children
                                team = "Away"
                            } else {
                                scorerInfo = event[0].children
                                team = "Home"
                            }
                            let scorer = scorerInfo[0].innerHTML.replace(/[\u200E]/g, "");;
                            let assister = scorerInfo[1];
                            if (assister) {
                                assister = assister.innerHTML.split("(Assist: ")[1].split(")")[0].replace(/[\u200E]/g, "");;
                            }
                            goalInfo[team].push({"goal": scorer, "assist": assister});
                        }
                    }
                }
            }
            return goalInfo;
        """)

        goal_data.append(goal_info)
        time.sleep(1)
    driver.quit()
    return goal_data

# Main function to run
if __name__ == "__main__":
    # Replace 'your_initial_url' with the actual URL you want to scrape
    initial_url = 'https://www.myfootballfacts.com/premier-league/all-time-premier-league/seasons/25-26/all-premier-league-matches-2025-26/'
    links = scrape_links(initial_url)
    # You might need to ensure full URLs if the links are relative
    base_url = 'your_base_url'
    full_links = [base_url + link if not link.startswith('http') else link for link in links]

    goal_information = extract_goal_info(full_links)
    print(goal_information)
    with open("2526Assists.json","w",encoding="utf-8") as f:
        string = json.dumps(goal_information)
        string = string.encode().decode("unicode_escape")
        f.write(string)