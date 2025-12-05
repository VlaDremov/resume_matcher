"""
LinkedIn Profile Scraper Module.

Provides optional functionality to scrape LinkedIn profile data using Selenium.

! Warning: Scraping LinkedIn violates their Terms of Service.
! Use the PDF export method (Profile.pdf) as the primary data source.
! This module is provided for educational purposes only.
"""

import json
import time
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# * Configuration
DEFAULT_WAIT_TIMEOUT = 10
SCROLL_PAUSE_TIME = 1.5


class LinkedInScraper:
    """
    Scraper for LinkedIn profile data.

    ! Important: This scraper requires manual login or pre-existing cookies.
    LinkedIn has strong anti-bot measures and will detect automated logins.
    """

    def __init__(self, headless: bool = False):
        """
        Initialize the LinkedIn scraper.

        Args:
            headless: Whether to run Chrome in headless mode.
                      Set to False for manual login.
        """
        self.driver: Optional[webdriver.Chrome] = None
        self.headless = headless

    def _init_driver(self) -> webdriver.Chrome:
        """Initialize Chrome WebDriver with appropriate options."""
        options = Options()

        if self.headless:
            options.add_argument("--headless=new")

        # * Standard options to appear more like a real browser
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # * Disable automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # * Remove webdriver property to avoid detection
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            },
        )

        return driver

    def start(self) -> None:
        """Start the browser session."""
        if self.driver is None:
            self.driver = self._init_driver()

    def stop(self) -> None:
        """Close the browser session."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def manual_login(self, timeout: int = 120) -> bool:
        """
        Open LinkedIn login page and wait for manual login.

        Args:
            timeout: Maximum time to wait for login (seconds).

        Returns:
            True if login was successful, False otherwise.
        """
        if self.driver is None:
            self.start()

        print("Opening LinkedIn login page...")
        print(f"Please log in manually within {timeout} seconds.")

        self.driver.get("https://www.linkedin.com/login")

        try:
            # * Wait for the feed page (indicates successful login)
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.ID, "global-nav"))
            )
            print("Login successful!")
            return True
        except Exception:
            print("! Login timed out or failed.")
            return False

    def save_cookies(self, filepath: str | Path) -> None:
        """
        Save current session cookies to a file.

        Args:
            filepath: Path to save cookies JSON file.
        """
        if self.driver is None:
            raise RuntimeError("Browser not started. Call start() first.")

        cookies = self.driver.get_cookies()
        filepath = Path(filepath)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2)

        print(f"Cookies saved to {filepath}")

    def load_cookies(self, filepath: str | Path) -> bool:
        """
        Load session cookies from a file.

        Args:
            filepath: Path to cookies JSON file.

        Returns:
            True if cookies were loaded successfully.
        """
        if self.driver is None:
            self.start()

        filepath = Path(filepath)

        if not filepath.exists():
            print(f"! Cookie file not found: {filepath}")
            return False

        # * First navigate to LinkedIn to set the domain
        self.driver.get("https://www.linkedin.com")

        with open(filepath, "r", encoding="utf-8") as f:
            cookies = json.load(f)

        for cookie in cookies:
            # * Remove problematic fields that might cause issues
            cookie.pop("sameSite", None)
            cookie.pop("storeId", None)

            try:
                self.driver.add_cookie(cookie)
            except Exception as e:
                print(f"? Warning: Could not add cookie: {e}")

        # * Refresh to apply cookies
        self.driver.refresh()

        return True

    def scrape_profile(self, profile_url: str) -> dict:
        """
        Scrape data from a LinkedIn profile.

        Args:
            profile_url: Full URL to the LinkedIn profile.

        Returns:
            Dictionary containing scraped profile data.
        """
        if self.driver is None:
            raise RuntimeError("Browser not started. Call start() first.")

        print(f"Navigating to profile: {profile_url}")
        self.driver.get(profile_url)

        # * Wait for profile to load
        try:
            WebDriverWait(self.driver, DEFAULT_WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.CLASS_NAME, "pv-top-card"))
            )
        except Exception:
            print("! Profile page did not load properly.")
            return {}

        # * Scroll down to load dynamic content
        self._scroll_page()

        profile_data = {
            "name": self._extract_name(),
            "headline": self._extract_headline(),
            "location": self._extract_location(),
            "about": self._extract_about(),
            "experience": self._extract_experience(),
            "education": self._extract_education(),
            "skills": self._extract_skills(),
        }

        return profile_data

    def _scroll_page(self) -> None:
        """Scroll down the page to load dynamic content."""
        if self.driver is None:
            return

        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # * Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            time.sleep(SCROLL_PAUSE_TIME)

            # * Calculate new height
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                break

            last_height = new_height

        # * Scroll back to top
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

    def _safe_find_text(self, by: str, value: str, default: str = "") -> str:
        """Safely find an element and return its text."""
        try:
            element = self.driver.find_element(by, value)
            return element.text.strip()
        except Exception:
            return default

    def _extract_name(self) -> str:
        """Extract profile name."""
        return self._safe_find_text(By.CSS_SELECTOR, "h1.text-heading-xlarge")

    def _extract_headline(self) -> str:
        """Extract profile headline."""
        return self._safe_find_text(By.CSS_SELECTOR, "div.text-body-medium")

    def _extract_location(self) -> str:
        """Extract profile location."""
        return self._safe_find_text(By.CSS_SELECTOR, "span.text-body-small.inline")

    def _extract_about(self) -> str:
        """Extract About section."""
        try:
            # * Click "see more" if present
            see_more_buttons = self.driver.find_elements(
                By.CSS_SELECTOR, "button.inline-show-more-text__button"
            )
            for btn in see_more_buttons:
                try:
                    btn.click()
                    time.sleep(0.3)
                except Exception:
                    pass

            about_section = self.driver.find_element(
                By.CSS_SELECTOR, "section.pv-about-section, div[class*='about']"
            )
            return about_section.text.strip()
        except Exception:
            return ""

    def _extract_experience(self) -> list[dict]:
        """Extract experience entries."""
        experience = []

        try:
            # * Find experience section
            exp_section = self.driver.find_element(By.ID, "experience")
            exp_container = exp_section.find_element(By.XPATH, "./following-sibling::div")

            # * Find all experience items
            exp_items = exp_container.find_elements(
                By.CSS_SELECTOR, "li.artdeco-list__item"
            )

            for item in exp_items[:10]:  # * Limit to first 10
                try:
                    title = item.find_element(
                        By.CSS_SELECTOR, "span[aria-hidden='true']"
                    ).text.strip()

                    company = ""
                    company_elements = item.find_elements(
                        By.CSS_SELECTOR, "span.t-14.t-normal"
                    )
                    if company_elements:
                        company = company_elements[0].text.strip()

                    experience.append({
                        "title": title,
                        "company": company,
                    })
                except Exception:
                    continue

        except Exception as e:
            print(f"? Could not extract experience: {e}")

        return experience

    def _extract_education(self) -> list[dict]:
        """Extract education entries."""
        education = []

        try:
            edu_section = self.driver.find_element(By.ID, "education")
            edu_container = edu_section.find_element(By.XPATH, "./following-sibling::div")

            edu_items = edu_container.find_elements(
                By.CSS_SELECTOR, "li.artdeco-list__item"
            )

            for item in edu_items[:5]:  # * Limit to first 5
                try:
                    institution = item.find_element(
                        By.CSS_SELECTOR, "span[aria-hidden='true']"
                    ).text.strip()

                    education.append({
                        "institution": institution,
                    })
                except Exception:
                    continue

        except Exception as e:
            print(f"? Could not extract education: {e}")

        return education

    def _extract_skills(self) -> list[str]:
        """Extract skills list."""
        skills = []

        try:
            skills_section = self.driver.find_element(By.ID, "skills")
            skills_container = skills_section.find_element(
                By.XPATH, "./following-sibling::div"
            )

            skill_elements = skills_container.find_elements(
                By.CSS_SELECTOR, "span[aria-hidden='true']"
            )

            for elem in skill_elements[:30]:  # * Limit to first 30
                skill_text = elem.text.strip()
                if skill_text and skill_text not in skills:
                    skills.append(skill_text)

        except Exception as e:
            print(f"? Could not extract skills: {e}")

        return skills

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def scrape_linkedin_profile(
    profile_url: str,
    cookies_path: Optional[str | Path] = None,
    save_cookies_path: Optional[str | Path] = None,
) -> dict:
    """
    Convenience function to scrape a LinkedIn profile.

    Args:
        profile_url: Full URL to the LinkedIn profile.
        cookies_path: Path to load existing cookies from.
        save_cookies_path: Path to save cookies after login.

    Returns:
        Dictionary containing scraped profile data.

    Example:
        >>> data = scrape_linkedin_profile(
        ...     "https://www.linkedin.com/in/vladislav-dremov/",
        ...     cookies_path="linkedin_cookies.json"
        ... )
    """
    with LinkedInScraper(headless=False) as scraper:
        if cookies_path and Path(cookies_path).exists():
            scraper.load_cookies(cookies_path)
        else:
            # * Manual login required
            if not scraper.manual_login():
                return {}

            if save_cookies_path:
                scraper.save_cookies(save_cookies_path)

        return scraper.scrape_profile(profile_url)

