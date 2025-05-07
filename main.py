from src.deepcrawl_chat.config import settings


def main():
    """
    testing function to run and test DeepCrawl Chat features.
    """
    print(settings.database.get_connection_string())
if __name__=="__main__":
    main()
    