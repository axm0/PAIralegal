import os
import time
import threading
from queue import Queue
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def click_begin_search(driver):
    try:
        begin_search_button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Begin Search')]"))
        )
        begin_search_button.click()
    except Exception as e:
        print(f"Error clicking Begin Search: {e}")

def accept_agreement(driver):
    try:
        accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Accept')]"))
        )
        accept_button.click()
    except Exception as e:
        print(f"Error accepting agreement: {e}")

def fill_search_form(driver, insurance_type):
    search_data = ""
    try:
        print(f"Starting to fill search form for {insurance_type}...")
        business_type_dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//label[@id='simpleSearch:businessType_label']"))
        )
        business_type_dropdown.click()
        prop_cas_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//li[contains(text(), 'Property & Casualty')]"))
        )
        prop_cas_option.click()
        search_data += "Business Type: Property & Casualty\n"
        type_of_insurance_dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//label[contains(@class, 'ui-selectcheckboxmenu-label')]"))
        )
        type_of_insurance_dropdown.click()
        time.sleep(2)
        insurance_type_xpath = f"//li[contains(@class, 'ui-selectcheckboxmenu-item')]//label[contains(text(), '{insurance_type}')]/preceding-sibling::div"
        insurance_type_checkbox = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, insurance_type_xpath))
        )
        insurance_type_checkbox.click()
        search_data += f"Type of Insurance: {insurance_type}\n"
        close_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'ui-selectcheckboxmenu-close')]"))
        )
        close_button.click()
        start_submission_date = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "simpleSearch:submissionStartDate_input"))
        )
        start_submission_date.clear()
        start_submission_date.send_keys("1/1/22")
        search_data += "Start Submission Date: 1/1/22\n"
        contains_label = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//label[text()='Contains']"))
        )
        driver.execute_script("arguments[0].click();", contains_label)
        search_data += "Company Name Search: Contains\n"
        company_name_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "simpleSearch:companyName"))
        )
        company_name_field.clear()
        company_name_field.send_keys("State Farm")
        search_data += "Company Name: State Farm\n"
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "simpleSearch:saveBtn"))
        )
        search_button.click()
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@id, 'filingTable')]"))
        )
        print(f"Results loaded successfully for {insurance_type}.")
    except Exception as e:
        print(f"Error filling out search form for {insurance_type}: {e}")
    finally:
        print(f"Current URL for {insurance_type}:", driver.current_url)
    return search_data

def process_search_results(driver):
    serff_numbers = []
    try:
        print("Processing search results...")
        table = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "j_idt25:filingTable"))
        )
        try:
            select = Select(driver.find_element(By.CSS_SELECTOR, "select.ui-paginator-rpp-options"))
            select.select_by_value('100')
            time.sleep(5)
        except Exception as e:
            print(f"Error changing rows per page: {e}")
        rows = driver.find_elements(By.CSS_SELECTOR, "#j_idt25\\:filingTable_data > tr")
        for i, row in enumerate(rows, start=1):
            try:
                serff_element = row.find_element(By.CSS_SELECTOR, f"td:nth-child(7)")
                serff_id = serff_element.text.split('-')[-1] if '-' in serff_element.text else serff_element.text
                serff_numbers.append(serff_id)
            except Exception as e:
                print(f"Error extracting SERFF ID from row {i}: {e}")
        print(f"Extracted {len(serff_numbers)} SERFF Tracking Numbers.")
    except Exception as e:
        print(f"Error processing search results: {e}")
        driver.save_screenshot("search_results_error.png")
        print("Screenshot saved as search_results_error.png")
    return serff_numbers

def process_filing(driver, serff_id, download_dir):
    filing_url = f"https://filingaccess.serff.com/sfa/search/filingSummary.xhtml?filingId={serff_id}"
    print(f"Processing SERFF ID: {serff_id}")
    driver.get(filing_url)
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'ui-panel-titlebar')]"))
    )
    time.sleep(5)
    summary_data = extract_filing_summary(driver)
    serff_folder = os.path.join(download_dir, f"SFMA-{serff_id}")
    os.makedirs(serff_folder, exist_ok=True)
    summary_file_path = os.path.join(serff_folder, f"{serff_id}_data.txt")
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(summary_data)
    print(f"Saved summary data to {summary_file_path}")
    buttons = [
        "formAttachmentSelectCurrentButton",
        "rateRuleAttachmentSelectCurrentButton",
        "supportingDocumentAttachmentSelectCurrentButton",
        "correspondenceAttachmentSelectAllButton"
    ]
    for button_id in buttons:
        try:
            button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, button_id)))
            driver.execute_script("arguments[0].click();", button)
        except Exception as e:
            print(f"Error clicking {button_id}: {e}")
    try:
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#summaryForm\\:downloadLink > span"))
        )
        driver.execute_script("arguments[0].click();", download_button)
        time.sleep(30)
        downloaded_file = max([f for f in os.listdir(download_dir) if f.endswith('.zip')],
                              key=lambda x: os.path.getctime(os.path.join(download_dir, x)))
        zip_file_path = os.path.join(serff_folder, f"SFMA-{serff_id}.zip")
        os.rename(os.path.join(download_dir, downloaded_file), zip_file_path)
        print(f"Moved zip file to {zip_file_path}")
    except Exception as e:
        print(f"Error downloading zip file: {e}")

def extract_filing_summary(driver):
    summary_data = ""
    try:
        summary_data += "Filing Information\n"
        filing_info = driver.find_element(By.XPATH, "//div[@class='col-lg-6'][1]")
        for row in filing_info.find_elements(By.XPATH, ".//div[@class='row']"):
            label = row.find_element(By.XPATH, ".//label").text.strip()
            value = row.find_element(By.XPATH, ".//div[contains(@class, 'col-sm-7')]").text.strip()
            summary_data += f"**{label}** {value}\n"
        summary_data += "\nFiling Outcome\n"
        filing_outcome = driver.find_element(By.XPATH, "//div[@class='col-lg-6'][2]")
        for row in filing_outcome.find_elements(By.XPATH, ".//div[@class='row']"):
            label = row.find_element(By.XPATH, ".//label").text.strip()
            value = row.find_element(By.XPATH, ".//div[contains(@class, 'col-sm-7')]").text.strip()
            summary_data += f"**{label}** {value}\n"
        summary_data += "\nCompany Information\n"
        company_info = driver.find_element(By.XPATH, "//div[@class='row' and ./label[contains(text(), 'Company Name')]]")
        headers = [header.text.strip() for header in company_info.find_elements(By.XPATH, ".//label")]
        summary_data += "**" + "**".join(headers) + "**\n"
        company_rows = driver.find_elements(By.XPATH, "//div[contains(@class, 'alternatingCompany') or contains(@style, 'margin-bottom:5px')]")
        for row in company_rows:
            columns = row.find_elements(By.XPATH, ".//span")
            row_data = " ".join([col.text.strip().replace("\n", " ") for col in columns])
            summary_data += row_data + "\n"
    except Exception as e:
        print(f"Error extracting filing summary: {e}")
    return summary_data

def scrape_insurance_type(insurance_type, base_dir):
    insurance_code = "4.0" if insurance_type == "04.0 Homeowners" else "19.0"
    insurance_name = "homeowners" if insurance_type == "04.0 Homeowners" else "personal_auto"
    folder_name = f"state_farm_{insurance_code}_{insurance_name}"
    download_dir = os.path.join(base_dir, folder_name)
    os.makedirs(download_dir, exist_ok=True)
    service = Service(r"C:\Developer\Drivers\chromedriver-win64\chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    try:
        url = "https://filingaccess.serff.com/sfa/home/CA" #change
        driver.get(url)
        driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
        driver.execute("send_command", params)
        click_begin_search(driver)
        accept_agreement(driver)
        search_data = fill_search_form(driver, insurance_type)
        metadata_file = os.path.join(download_dir, f"{insurance_name}_metadata.txt")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(search_data)
        print(f"Saved metadata to {metadata_file}")
        serff_numbers = process_search_results(driver)
        print(f"\nTotal SERFF numbers extracted for {insurance_type}: {len(serff_numbers)}")
        for serff_id in serff_numbers:
            process_filing(driver, serff_id, download_dir)
        print(f"All filings have been processed for {insurance_type}.")
    except Exception as e:
        print(f"Error in scraping process for {insurance_type}: {e}")
    finally:
        driver.quit()

def scrape_state_farm_filings():
    base_dir = r"C:\Developer\Workspace\llama3.2\data"
    os.makedirs(base_dir, exist_ok=True)
    insurance_types = ["04.0 Homeowners", "19.0 Personal Auto"]
    threads = []
    for insurance_type in insurance_types:
        thread = threading.Thread(target=scrape_insurance_type, args=(insurance_type, base_dir))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print("All insurance types have been processed.")

if __name__ == "__main__":
    scrape_state_farm_filings()
