import requests
from pathlib import Path

urls = {
    "luat-phong-chong-ma-tuy-2021.pdf": "https://pbgdpl.haiphong.gov.vn/Uploads/2021/11/luat-phong-chong-ma-tuy-2021.pdf",
    "nghi-dinh-105-2021.pdf": "http://casocai-nghienmatuy.khanhhoa.gov.vn/Uploads/Images/files/T%C3%A0i%20li%E1%BB%87u%20tuy%C3%AAn%20truy%E1%BB%81n/Ngh%E1%BB%8B%20%C4%91%E1%BB%8Bnh%20105.pdf",
    "bo-luat-hinh-su-2015.pdf": "https://sgddt.tiangiang.gov.vn/documents/1031310/21576426/bo_luat_hinh_su_2015_sua_doi_2017.pdf"
}

for name, url in urls.items():
    print(f"Downloading {name} from {url}...")
    try:
        r = requests.get(url, timeout=15, verify=False)
        if r.status_code == 200:
            print(f"Success! {len(r.content)} bytes")
        else:
            print(f"Failed status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")
