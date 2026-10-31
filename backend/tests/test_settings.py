from settings import sync_url


def test_sync_url_swaps_asyncmy_for_pymysql():
    assert sync_url("mysql+asyncmy://u:p@h:3306/db") == "mysql+pymysql://u:p@h:3306/db"


def test_sync_url_leaves_sync_urls_alone():
    assert sync_url("sqlite:///x.db") == "sqlite:///x.db"
