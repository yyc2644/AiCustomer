"""
Page Object基类模块
提供页面对象模式的基础类

功能：
1. 统一的元素定位方法
2. 等待机制
3. 截图功能
4. 日志记录

使用方法：
    from lib.page_objects.base import BasePage, Locator
    
    class ChatPage(BasePage):
        def __init__(self, driver):
            super().__init__(driver)
            self.input_box = Locator("css", "textarea[placeholder*='请输入']")
            self.send_button = Locator("css", "button.send-btn")
        
        def send_message(self, message):
            self.type(self.input_box, message)
            self.click(self.send_button)
"""

import time
import os
import logging
from typing import Optional, Tuple, List, Any, Dict
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class Locator:
    """元素定位器"""
    type: str          # 定位方式: css, xpath, id, name, class_name, etc.
    value: str         # 定位值
    description: str = ""  # 描述
    
    def __str__(self):
        return f"Locator({self.type}={self.value})"


class BasePage:
    """页面对象基类"""
    
    def __init__(self, driver, locator_file: str = None):
        """
        初始化页面对象
        
        Args:
            driver: WebDriver实例
            locator_file: 定位器文件路径（可选）
        """
        self.driver = driver
        self.locators: Dict[str, Locator] = {}
        self.timeout = 10  # 默认超时时间
        self.screenshot_dir = "reports/screenshots"
        
        # 加载定位器
        if locator_file:
            self._load_locators(locator_file)
    
    def _load_locators(self, locator_file: str):
        """加载定位器文件"""
        try:
            from data.data_loader import DataLoader
            
            locators_data = DataLoader.load_yaml(locator_file)
            
            if 'locators' in locators_data:
                for name, loc in locators_data['locators'].items():
                    self.locators[name] = Locator(
                        type=loc.get('type', 'css'),
                        value=loc.get('value', ''),
                        description=loc.get('description', '')
                    )
            
            logger.info(f"加载定位器: {len(self.locators)} 个")
            
        except Exception as e:
            logger.warning(f"加载定位器失败: {e}")
    
    def get_locator(self, name: str) -> Optional[Locator]:
        """获取定位器"""
        return self.locators.get(name)
    
    def add_locator(self, name: str, locator: Locator):
        """添加定位器"""
        self.locators[name] = locator
    
    # ============================================
    # 元素查找
    # ============================================
    
    def find_element(self, locator: Locator, timeout: int = None) -> Any:
        """
        查找单个元素
        
        Args:
            locator: 定位器
            timeout: 超时时间
            
        Returns:
            WebElement
            
        Raises:
            TimeoutException: 查找超时
        """
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                element = self._find_by_locator(locator)
                if element:
                    return element
            except Exception as e:
                logger.debug(f"查找元素 {locator} 失败: {e}")
            
            time.sleep(0.5)
        
        raise TimeoutError(f"查找元素超时: {locator}")
    
    def find_elements(self, locator: Locator) -> List[Any]:
        """查找多个元素"""
        return self._find_by_locator(locator, multiple=True)
    
    def _find_by_locator(self, locator: Locator, multiple: bool = False) -> Any:
        """根据定位器查找元素"""
        by = locator.type.lower()
        value = locator.value
        
        if multiple:
            if by == "css":
                return self.driver.find_elements("css selector", value)
            elif by == "xpath":
                return self.driver.find_elements("xpath", value)
            elif by == "id":
                return self.driver.find_elements("id", value)
            elif by == "name":
                return self.driver.find_elements("name", value)
            elif by == "class":
                return self.driver.find_elements("class name", value)
            else:
                return self.driver.find_elements(by, value)
        else:
            if by == "css":
                return self.driver.find_element("css selector", value)
            elif by == "xpath":
                return self.driver.find_element("xpath", value)
            elif by == "id":
                return self.driver.find_element("id", value)
            elif by == "name":
                return self.driver.find_element("name", value)
            elif by == "class":
                return self.driver.find_element("class name", value)
            else:
                return self.driver.find_element(by, value)
    
    def find_element_by_name(self, name: str, timeout: int = None) -> Any:
        """通过名称查找元素"""
        locator = self.get_locator(name)
        if not locator:
            raise ValueError(f"定位器未找到: {name}")
        return self.find_element(locator, timeout)
    
    # ============================================
    # 元素操作
    # ============================================
    
    def click(self, locator: Locator):
        """点击元素"""
        element = self.find_element(locator)
        element.click()
        logger.debug(f"点击元素: {locator}")
    
    def type(self, locator: Locator, text: str, clear: bool = True):
        """输入文本"""
        element = self.find_element(locator)
        if clear:
            element.clear()
        element.send_keys(text)
        logger.debug(f"输入文本: {locator} -> {text}")
    
    def get_text(self, locator: Locator) -> str:
        """获取元素文本"""
        element = self.find_element(locator)
        return element.text
    
    def get_attribute(self, locator: Locator, attr: str) -> str:
        """获取元素属性"""
        element = self.find_element(locator)
        return element.get_attribute(attr)
    
    def is_visible(self, locator: Locator) -> bool:
        """检查元素是否可见"""
        try:
            element = self.find_element(locator, timeout=2)
            return element.is_displayed()
        except:
            return False
    
    def is_enabled(self, locator: Locator) -> bool:
        """检查元素是否可用"""
        try:
            element = self.find_element(locator, timeout=2)
            return element.is_enabled()
        except:
            return False
    
    def wait_for_visible(self, locator: Locator, timeout: int = None) -> bool:
        """等待元素可见"""
        try:
            self.find_element(locator, timeout)
            return True
        except:
            return False
    
    def wait_for_invisible(self, locator: Locator, timeout: int = None) -> bool:
        """等待元素消失"""
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.is_visible(locator):
                return True
            time.sleep(0.5)
        
        return False
    
    # ============================================
    # 页面操作
    # ============================================
    
    def navigate_to(self, url: str):
        """导航到URL"""
        self.driver.get(url)
        logger.info(f"导航到: {url}")
    
    def get_current_url(self) -> str:
        """获取当前URL"""
        return self.driver.current_url
    
    def refresh(self):
        """刷新页面"""
        self.driver.refresh()
    
    def go_back(self):
        """返回上一页"""
        self.driver.back()
    
    def go_forward(self):
        """前进一页"""
        self.driver.forward()
    
    # ============================================
    # 截图
    # ============================================
    
    def take_screenshot(self, name: str = None) -> str:
        """
        截图
        
        Args:
            name: 文件名（不含扩展名）
            
        Returns:
            截图文件路径
        """
        # 确保目录存在
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # 生成文件名
        if not name:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_path = os.path.join(self.screenshot_dir, f"{name}.png")
        
        # 截图
        self.driver.save_screenshot(file_path)
        logger.info(f"截图保存到: {file_path}")
        
        return file_path
    
    def take_element_screenshot(self, locator: Locator, name: str = None) -> str:
        """元素截图"""
        element = self.find_element(locator)
        
        # 生成文件名
        if not name:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_path = os.path.join(self.screenshot_dir, f"{name}.png")
        
        # 需要通过JavaScript截图（Playwright支持）
        try:
            self.driver.save_screenshot(file_path)
        except:
            # 备选：全屏截图
            self.take_screenshot(name)
        
        return file_path
    
    # ============================================
    # 等待和轮询
    # ============================================
    
    def wait(self, seconds: float):
        """固定等待"""
        time.sleep(seconds)
    
    def wait_for_condition(self, condition: callable, timeout: int = None,
                          interval: float = 0.5) -> bool:
        """
        等待条件满足
        
        Args:
            condition: 条件函数
            timeout: 超时时间
            interval: 检查间隔
            
        Returns:
            条件是否满足
        """
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if condition():
                    return True
            except:
                pass
            
            time.sleep(interval)
        
        return False
    
    # ============================================
    # 工具方法
    # ============================================
    
    def execute_script(self, script: str, *args):
        """执行JavaScript"""
        return self.driver.execute_script(script, *args)
    
    def switch_to_frame(self, locator: Locator = None):
        """切换到iframe"""
        if locator:
            frame = self.find_element(locator)
            self.driver.switch_to.frame(frame)
        else:
            self.driver.switch_to.default_content()
    
    def switch_to_window(self, window_index: int = 0):
        """切换窗口"""
        windows = self.driver.window_handles
        if window_index < len(windows):
            self.driver.switch_to.window(windows[window_index])
    
    def close(self):
        """关闭当前窗口"""
        self.driver.close()
    
    def quit(self):
        """退出浏览器"""
        self.driver.quit()


class LocatorHelper:
    """定位器辅助类"""
    
    @staticmethod
    def css(selector: str, description: str = "") -> Locator:
        """CSS选择器定位器"""
        return Locator("css", selector, description)
    
    @staticmethod
    def xpath(xpath: str, description: str = "") -> Locator:
        """XPath定位器"""
        return Locator("xpath", xpath, description)
    
    @staticmethod
    def id(element_id: str, description: str = "") -> Locator:
        """ID定位器"""
        return Locator("id", element_id, description)
    
    @staticmethod
    def name(name: str, description: str = "") -> Locator:
        """Name定位器"""
        return Locator("name", name, description)
    
    @staticmethod
    def class_name(class_name: str, description: str = "") -> Locator:
        """Class名称定位器"""
        return Locator("class", class_name, description)
    
    @staticmethod
    def text(text: str, description: str = "") -> Locator:
        """文本定位器（通过XPath）"""
        return Locator("xpath", f"//*[text()='{text}']", description)
    
    @staticmethod
    def contains(text: str, description: str = "") -> Locator:
        """包含文本定位器"""
        return Locator("xpath", f"//*[contains(text(),'{text}')]", description)


# 便捷函数
def css(selector: str, description: str = "") -> Locator:
    return LocatorHelper.css(selector, description)

def xpath(xpath: str, description: str = "") -> Locator:
    return LocatorHelper.xpath(xpath, description)

def id(element_id: str, description: str = "") -> Locator:
    return LocatorHelper.id(element_id, description)

def name(name: str, description: str = "") -> Locator:
    return LocatorHelper.name(name, description)
