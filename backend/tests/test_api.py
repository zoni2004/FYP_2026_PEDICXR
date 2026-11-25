from conftest import xray_png


def upload(client, prob_fake=None, fake=None, name="xray.png", data=None, age=28, sex="M", ref="Child 0001"):
    if fake is not None and prob_fake is not None:
        fake.next_probability = prob_fake
    return client.post("/api/studies",
                       files={"image": (name, data if data is not None else xray_png(), "image/png")},
                       data={"age_months": str(age), "sex": sex, "patient_ref": ref})


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model": "fake-model"}


def test_create_study_returns_result_heatmap_and_draft(client, fake):
    r = upload(client, 0.8, fake)
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["code"].startswith("PCX-")
    assert s["status"] == "review"
    assert s["findings"] == [{"name": "Pneumonia", "probability": 0.8, "positive": True}]
    assert s["pneumonia_probability"] == 0.8
    assert s["report"]["status"] == "draft"
    assert "pneumonia" in s["report"]["text"].lower()
    assert "right" in s["report"]["text"]          # peak x=0.3 is the patient's right
    img = client.get(s["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    heat = client.get(s["heatmap_url"])
    assert heat.status_code == 200 and heat.content[:4] == b"\x89PNG"


def test_low_probability_gives_normal_report(client, fake):
    s = upload(client, 0.1, fake).json()
    assert s["findings"][0]["positive"] is False
    assert "no radiographic evidence of pneumonia" in s["report"]["text"].lower()


def test_rejects_wrong_file_type(client):
    r = upload(client, name="notes.txt", data=b"hello")
    assert r.status_code == 415
    assert "PNG, JPG or DICOM" in r.json()["detail"]


def test_rejects_broken_image(client):
    r = upload(client, name="broken.png", data=b"not really a png")
    assert r.status_code == 400


def test_rejects_bad_age_and_sex(client):
    assert upload(client, age=300).status_code == 422
    assert upload(client, sex="X").status_code == 422


def test_list_sorted_and_filtered(client, fake):
    upload(client, 0.95, fake, ref="High one")
    upload(client, 0.05, fake, ref="Low one")
    rows = client.get("/api/studies", params={"status": "review"}).json()
    probs = [r["pneumonia_probability"] for r in rows]
    assert probs == sorted(probs, reverse=True)
    found = client.get("/api/studies", params={"q": "high one"}).json()
    assert [r["patient_ref"] for r in found] == ["High one"]


def test_report_draft_then_approve_then_locked(client, fake):
    sid = upload(client, 0.7, fake).json()["id"]
    r = client.put(f"/api/studies/{sid}/report", json={"text": "Edited draft", "status": "draft"})
    assert r.status_code == 200 and r.json()["report"]["text"] == "Edited draft"
    assert r.json()["status"] == "review"
    r = client.put(f"/api/studies/{sid}/report", json={"text": "Final text", "status": "approved"})
    assert r.json()["status"] == "approved" and r.json()["report"]["status"] == "approved"
    assert client.put(f"/api/studies/{sid}/report", json={"text": "again"}).status_code == 409
    assert sid in [s["id"] for s in client.get("/api/studies", params={"status": "approved"}).json()]


def test_feedback(client, fake):
    sid = upload(client, 0.6, fake).json()["id"]
    assert client.post(f"/api/studies/{sid}/feedback", json={"finding": "Pneumonia", "agree": False}).status_code == 201
    assert client.get(f"/api/studies/{sid}").json()["feedback"] == {"Pneumonia": False}
    assert client.post(f"/api/studies/{sid}/feedback", json={"finding": "Effusion", "agree": True}).status_code == 400


def test_pdf(client, fake):
    sid = upload(client, 0.8, fake).json()["id"]
    r = client.get(f"/api/studies/{sid}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_model_info(client):
    m = client.get("/api/model-info").json()
    assert m["name"] == "fake-model"
    assert abs(m["test_metrics"]["auc"]["mean"] - 0.82) < 0.01
    assert len(m["folds"]) == 5


def test_unknown_study_is_404(client):
    assert client.get("/api/studies/999999").status_code == 404


def test_database_file_is_not_served(client):
    assert client.get("/files/db/pedicxr.db").status_code == 404
    assert client.get("/files/uploads/../db/pedicxr.db").status_code == 404
