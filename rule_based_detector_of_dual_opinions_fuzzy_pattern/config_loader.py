import json

class ConfigLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config_data = None

    def load_config(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.config_data = json.load(file)

    def get_value(self, key):
        if self.config_data:
            return self.config_data.get(key)
        else:
            raise ValueError("Config file not loaded.")

    def get_values(self):
        if self.config_data:
            return self.config_data
        else:
            raise ValueError("Config file not loaded.")

    def save_threshold(self, threshold):
        self.config_data["threshold"] = threshold

        with open("config.json", "w") as file:
            json.dump(self.config_data, file, indent=4)