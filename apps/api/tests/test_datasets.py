from fastapi.testclient import TestClient
from app import app  # Assuming your FastAPI application is named app
from models.base import Dataset
from models.controller import dataset_manager
import time

client = TestClient(app)

def test_create_dataset():
    """
    Tests the POST /v1/datasets endpoint.
    The endpoint is supposed to create a new dataset.
    """
    response = client.post("/v1/datasets", json={"documents": []})

    # the endpoint should return with a 200 OK status and the id of the created dataset
    assert response.status_code == 200
    assert "id" in response.json()["data"]

def test_delete_dataset():
    """
    Tests the DELETE /v1/datasets/{id} endpoint.
    The endpoint is supposed to delete an existing dataset.
    """
    # delete the dataset created in the test_create_dataset test
    response = client.delete("/v1/datasets/test1")

    # the endpoint should return with a 200 OK status
    assert response.status_code == 200

def test_get_dataset():
    """
    Tests the GET /v1/datasets/{id} endpoint.
    The endpoint is supposed to return a specific dataset given its id.
    """
    # insert a dataset into the system for testing
    test_dataset = Dataset(
        id="test1",
        documents=[
            {
                "url": "https://storage.googleapis.com/context-builder/public-tmp/0wpj5TqcnFRM.pdf",
                "type": "pdf",
                "uid": "IQtABa9mZxfm",
            }
        ]
    )
    dataset_manager.save_dataset(test_dataset)

    response = client.get("/v1/datasets/test1")

    # the endpoint should return with a 200 OK status
    assert response.status_code == 200
    dataset_manager.delete_dataset('test1')

def test_update_dataset_with_annotated_data():
    """
    Tests the PATCH /v1/datasets/{id} endpoint with annotated data.
    """
    # create a dataset for the test
    test_dataset_id = 'test_update_annotated'
    test_document_uid = 'bdc2414eeeda4f84b5acf94e28b2c2ec'
    dataset_manager.delete_dataset(test_dataset_id)
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[]
    )
    dataset_manager.save_dataset(test_dataset)
    update_data = {
        "documents": [
            {
                "uid": test_document_uid,
                "url": test_document_uid,
                "type": "annotated_data",
                "split_option": {
                    "split_type": "character",
                    "chunk_size": 20,
                    "chunk_overlap": 10
                }
            }
        ]
    }
    # send the PATCH request
    response = client.patch(
        f"/v1/datasets/{test_dataset_id}",
        json=update_data
    )
    assert response.status_code == 200
    # wait for the asyc task complete and then fetch the updated dataset to verify the changes
    time.sleep(30)
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data['message'] == 'success'
    assert response_data['data']['totalItems'] == 7
    # Verify the first three segments' content and IDs
    expected_segments = [
        {"segment_id": f"{test_dataset_id}-{test_document_uid}-0", "content": "Human:adadadad"},
        {"segment_id": f"{test_dataset_id}-{test_document_uid}-1", "content": "AI:Introduce yourself."},
        {"segment_id": f"{test_dataset_id}-{test_document_uid}-2", "content": "Human:谢谢你因为有你"},
    ]
    for expected_segment in expected_segments:
        segment = next((seg for seg in response_data['data']['segments'] if seg['segment_id'] == expected_segment['segment_id']), None)
        assert segment is not None, f"Segment {expected_segment['segment_id']} not found."
        assert segment['content'] == expected_segment['content'], f"Content for segment {expected_segment['segment_id']} does not match."
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)
    
def test_update_dataset():
    """
    Tests the PATCH /v1/datasets/{id} endpoint.
    The endpoint is supposed to update an existing dataset, first with two documents, then with one.
    """
    test_dataset_id = 'test_update'
    dataset_manager.delete_dataset(test_dataset_id)
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[]
    )
    dataset_manager.save_dataset(test_dataset)
    # update with two documents
    response_full_update = client.patch(
        f"/v1/datasets/{test_dataset_id}",
        json={
            "documents": [
                {
                    "uid": "9qFAhcql1RmA",
                    "url": "https://storage.googleapis.com/context-builder/public-tmp/CjefkDt4uecw.doc",
                    "type": "word",
                    "split_option": {
                        "split_type": "character",
                        "chunk_size": 500,
                        "chunk_overlap": 0
                    }
                },
                {
                    "uid": "UGrEk6nGof9G",
                    "url": "https://storage.googleapis.com/context-builder/public-tmp/J6D08G9I5ja0.pdf",
                    "type": "pdf",
                    "split_option": {
                        "split_type": "character",
                        "chunk_size": 500,
                        "chunk_overlap": 0
                    }
                }
            ]
        }
    )
    assert response_full_update.status_code == 200
    time.sleep(40)
    response = client.get(f"/v1/datasets/{test_dataset_id}")
    dataset_data = response.json()['data'][0]
    assert len(dataset_data['documents']) == 2
    # only with the first document, the second document should be deleted by this operation
    response_partial_update = client.patch(
        f"/v1/datasets/{test_dataset_id}",
        json={
            "documents": [
                {
                    "uid": "9qFAhcql1RmA",
                    "url": "https://storage.googleapis.com/context-builder/public-tmp/CjefkDt4uecw.doc",
                    "type": "word",
                    "split_option": {
                        "split_type": "character",
                        "chunk_size": 500,
                        "chunk_overlap": 0
                    }
                }
            ]
        }
    )

    assert response_partial_update.status_code == 200
    # fetch the updated dataset and check if only one document remains
    time.sleep(30)
    response = client.get(f"/v1/datasets/{test_dataset_id}")
    dataset_data = response.json()['data'][0]
    assert len(dataset_data['documents']) == 1
    assert dataset_data['documents'][0]['uid'] == "9qFAhcql1RmA"
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)

def test_update_dataset_preview():
    """
    Tests the PATCH /v1/datasets/{id} endpoint's preview functionality.
    It should update a dataset preview given the preview count and UID of the document.
    """
    # create a test dataset
    test_dataset_id = 'test_update_preview'
    test_document_uid = "9qFAhcql1RmA"
    dataset_manager.delete_dataset(test_dataset_id)
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[]
    )
    dataset_manager.save_dataset(test_dataset)
    # define the dataset with a preview document
    update_data = {
        "documents": [
            {
            "uid": test_document_uid,
            "url": "https://storage.googleapis.com/context-builder/public-tmp/CjefkDt4uecw.doc",
            "type": "word",
            "split_option": {
                "split_type": "character",
                "chunk_size": 600,
                "chunk_overlap": 0
            }
            }
        ]
    }
    # send the PATCH request with the preview parameter
    response = client.patch(
        f"/v1/datasets/{test_dataset_id}?preview=5&uid={test_document_uid}",
        json=update_data
    )
    # assert the request was successful
    assert response.status_code == 200
    response_data = response.json()
    assert response_data['message'] == 'success'
    assert response_data['status'] == 200
    # may verify the preview content afterwards if needed
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)

def test_retrieve_document_segments():
    test_dataset_id = 'test_update'
    dataset_manager.delete_dataset(test_dataset_id)
    test_document_uid = "FKuvmEac1CGr"
    test_document_url = "https://storage.googleapis.com/context-builder/public-tmp/J6D08G9I5ja0.pdf"
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[
            {
                "uid": test_document_uid,
                "url": "https://storage.googleapis.com/context-builder/public-tmp/J6D08G9I5ja0.pdf",
                "type": "pdf",
                "split_option": {
                    "split_type": "character",
                    "chunk_size": 500,
                    "chunk_overlap": 0
                }
            }
        ]
    )
    dataset_manager.save_dataset(test_dataset)
    expected_segments = [
        {"segment_id": f"{test_dataset_id}-{test_document_url}-0", "content":"婚俗新风蔚然兴起  \n2023年09月26日 06:00  来源：光明日报大字体小字体  \n  美好的婚俗新风，正在全国各地蔚然兴起。  \n \n  “最美婚姻登记处即将启用！”这个消息刚刚 在“舟山普陀文旅”微信公众号上发布，便刷爆\n了当地人的朋友圈。  \n \n  “我们将新的婚姻登记管理服务中心设在海滨公园，占地面积 8000平方米，专门设计了爱\n情主题公园、海岛婚俗体验馆等，为婚姻登记赋予独特韵味。”浙江省舟山市普陀区民政局局\n长丁勇说。  \n \n  “在这里领证，仪式感瞬间拉满！”一对新人手持结婚证，与记者分享喜悦。  \n \n  普陀着眼地域特点和风土人情，推进婚俗改革，打造了白沙岛户外颁证基地、最美爱情公\n路等约会地，吸引了越来越多的年轻人来打卡。  \n \n  “我们自愿结为夫妻，一起肩负婚姻赋予的所有责任，所有义务… …”伴随着浪漫的音乐，\n在广东广州从化区生态设计小镇的草坪上，从化区民政局举办了一场别开生面的“花城有囍”集\n体婚礼。 6对新人许下结婚誓言，接过结婚证书，掀开人生的新篇章。"},
        {"segment_id": f"{test_dataset_id}-{test_document_url}-1", "content":"“蓝天白云下，亲朋好友见证嘉礼，别有一番情调。户外颁证形式虽简单，但仪式感十\n足，我们希望和所有人分享这份美好。”一对新人兴奋地向其他新人和嘉宾派发喜糖。  \n \n  广州市民政局社会事务处副处长许婵娟告诉记者：“近年来，我们举办了 100多场集体婚\n礼和集体户外颁证活动，累计 9000多位新人及亲友参加，市民对‘小型婚礼、大美婚俗’的认同\n度不断提高。”  \n \n  广州在全市传统文化街区、公园景区、城市广场和网红打卡点打造“花城有囍”户外颁证\n点，目前共有 34个室内颁证厅和形式多样的“花城有囍”户外颁证点投入使用，并建立特邀颁\n证嘉宾库，提供结婚颁证远程观礼和特邀颁证服务，弘扬文明健康、低碳简约的现代婚俗新\n风，受到越来越多市民青睐。  \n \n  在山东青岛西海岸新区铁山街道别家网格村，村民都说现在娶亲有了“新底气”。  \n \n  原来，别家网格村村民代表大会制定了《别家网格婚事操办标准》 ，对婚车、宴席、礼金\n都进行了约定，同时投资 87万元建设婚礼堂，免费供村民办婚礼使用，村集体和帮 办服务队\n免费提供场地、厨师、司仪等。这样一来，群众举办婚礼费用大大降低。"},
        {"segment_id": f"{test_dataset_id}-{test_document_url}-2", "content":"“前段时间我家孩子结婚，我和亲家决定办喜事一切从简，不仅省心省钱，还登上了村里\n移风易俗光荣榜，这可比大操大办更有面子！”别家网格村村民王秀花说。  \n \n  “我们通过调研发现，大多数群众对高价彩礼、铺张浪费、随礼攀比等不正之风非常抵\n制，婚俗新风很受欢迎。”青岛西海岸新区民政局局长马志广介绍，新区指导村（社区）将喜\n事新办简办纳入村规民约，制定群众认可的具有约束性的红事操办标准，成立村（社区）喜事\n帮办服务队，通过协助操办红事促进形成文明 新风。  \n \n  婚俗新风正吹过每一个角落。河北省河间市邀请市领导及市人大代表、政协委员、道德模\n范等为“零彩礼”“低彩礼”新人颁发结婚证， 2021年以来共为 5947对新人办理结婚登记；四川\n省成都市武侯区在社区内建设了多民族婚俗微型博物馆，以多民族婚俗文化为主题，展示弘扬\n中华优秀传统婚俗文化，引领婚俗新风向上向善；湖南省澧县对参加集体婚礼后不再办婚宴和\n不要彩礼的新人，由县政府给予一定的奖励，并以新人名义将奖金捐献给属地公益事业。  \n \n  “下一步，各级民政部门将持续推进婚俗领域移风易俗，治理高价彩礼、大操大办、随礼"},
        {"segment_id": f"{test_dataset_id}-{test_document_url}-3", "content":"攀比等婚嫁陋习，积极创新婚俗文化载体，教育引导青年树立正确的婚恋观、家庭观，构建新\n型婚育文化，助力婚姻家庭幸福稳定。”民政部社会事务司司长王金华说。"}
    ]
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data['data']['totalItems'] == 4
    # check segment_id
    for i, segment in enumerate(response_data['data']['segments']):
        assert segment['segment_id'] == expected_segments[i]['segment_id']
    # check content
    for i, segment in enumerate(response_data['data']['segments']):
        assert segment['content'] == expected_segments[i]['content']
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)

def test_retrieve_document_segments_with_query():
    """
    Tests the GET /v1/datasets/{dataset_id}/document/{uid} endpoint with a search query.
    It should return segments that contain the search query.
    """
    test_dataset_id = 'test_query'
    test_document_uid = "FKuvmEac1CGr"
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[
            {
                "uid": test_document_uid,
                "url": "https://storage.googleapis.com/context-builder/public-tmp/J6D08G9I5ja0.pdf",
                "type": "pdf",
                "split_option": {
                    "split_type": "character",
                    "chunk_size": 500,
                    "chunk_overlap": 0
                }
            }
        ]
    )
    dataset_manager.save_dataset(test_dataset)
    # retrieve segments with a search query
    query_string = "为"
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}?query={query_string}")
    assert response.status_code == 200
    response_data = response.json()
    # check that each returned segment contains the query string '为'
    for segment in response_data['data']['segments']:
        assert query_string in segment['content'], f"The segment content does not contain the query string '{query_string}'."
    # check the totalItems
    expected_number_of_segments_with_query = 2  # Replace with the expected number based on your test data
    assert response_data['data']['totalItems'] == expected_number_of_segments_with_query, \
        "The number of segments containing the query string does not match the expected count."
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)

def test_add_segments():
    test_dataset_id = 'test_add_segment'
    test_document_uid = "FKuvmEac1CGr"
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[
            {
                "uid": test_document_uid,
                "url": "https://storage.googleapis.com/context-builder/public-tmp/J6D08G9I5ja0.pdf",
                "type": "pdf",
                "split_option": {
                    "split_type": "character",
                    "chunk_size": 500,
                    "chunk_overlap": 0
                }
            }
        ]
    )
    dataset_manager.save_dataset(test_dataset)
    # add segments
    new_segment_content = "新添加的段落内容。"
    response = client.post(
        f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}/segment/",
        json={"content": new_segment_content}
    )
    assert response.status_code == 200
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    assert response.status_code == 200
    response_data = response.json()
    # check value of total items
    assert response_data['data']['totalItems'] == 5, f"Expected totalItems to be 5, got {response_data['data']['totalItems']}"
    # check the content of the last segment
    last_segment = response_data['data']['segments'][-1]
    assert last_segment['content'] == new_segment_content, "The content of the last segment does not match the new segment content."
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)

def test_edit_segment():
    test_dataset_id = 'test_edit_segment'
    test_document_uid = "FKuvmEac1CGr"
    test_document_url = "https://storage.googleapis.com/context-builder/public-tmp/J6D08G9I5ja0.pdf"
    test_segment_id = f"{test_dataset_id}-{test_document_url}-0"
    dataset_manager.delete_dataset(test_dataset_id)
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[
            {
                "uid": test_document_uid,
                "url": test_document_url,
                "type": "pdf",
                "split_option": {
                    "split_type": "character",
                    "chunk_size": 500,
                    "chunk_overlap": 0
                }
            }
        ]
    )
    dataset_manager.save_dataset(test_dataset)
    updated_content = "这是更新后的段落内容。"
    # edit segment
    response = client.patch(
        f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}/segment/{test_segment_id}",
        json={"content": updated_content}
    )
    assert response.status_code == 200
    # check whether update successfully
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    assert response.status_code == 200
    response_data = response.json()
    segment = next((s for s in response_data['data']['segments'] if s['segment_id'] == test_segment_id), None)
    assert segment is not None, "The edited segment was not found."
    assert segment['content'] == updated_content, "The content of the segment did not match the updated content."
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)

def test_delete_segment():
    test_dataset_id = 'test_delete_segment'
    test_document_uid = "FKuvmEac1CGr"
    test_document_url = "https://storage.googleapis.com/context-builder/public-tmp/J6D08G9I5ja0.pdf"
    test_segment_id = f"{test_dataset_id}-{test_document_url}-0"
    dataset_manager.delete_dataset(test_dataset_id)
    test_dataset = Dataset(
        id=test_dataset_id,
        documents=[
            {
                "uid": test_document_uid,
                "url": test_document_url,
                "type": "pdf",
                "split_option": {
                    "split_type": "character",
                    "chunk_size": 500,
                    "chunk_overlap": 0
                }
            }
        ]
    )
    dataset_manager.save_dataset(test_dataset)
    # get current items number
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    original_data = response.json()
    original_total_items = original_data['data']['totalItems']
    # delete segment
    response = client.patch(
        f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}/segment/{test_segment_id}",
        json={"content": ""}
    )
    assert response.status_code == 200
    # check whether delete successfully
    time.sleep(30)
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data['data']['totalItems'] == original_total_items - 1, "The totalItems did not decrease after deletion."
    assert not any(s['segment_id'] == test_segment_id for s in response_data['data']['segments']), "The deleted segment was found."
    # cleanup
    dataset_manager.delete_dataset(test_dataset_id)

def test_dataset_integration():
    """
    Integration test to cover the process of creating, updating, adding segments to,
    editing, and deleting from a dataset.
    """
    # create a new dataset
    response = client.post("/v1/datasets", json={"documents": []})
    assert response.status_code == 200
    data = response.json()
    test_dataset_id = data["data"]["id"]

    # ppdate the dataset by adding a new document
    test_document_uid = "test_document_uid"
    test_document_url = "https://storage.googleapis.com/context-builder/public-tmp/vjn5GCBpSsKl.pdf"
    update_data = {
        "documents": [
            {
                "uid": test_document_uid,
                "url": test_document_url,
                "type": "pdf",
                "split_option": {
                    "split_type": "character",
                    "chunk_size": 500,
                    "chunk_overlap": 10
                }
            }
        ]
    }
    response = client.patch(f"/v1/datasets/{test_dataset_id}", json=update_data)
    assert response.status_code == 200
    time.sleep(25)
    # add a new segment to the document
    new_segment_content = "New segment content"
    response = client.post(
        f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}/segment/",
        json={"content": new_segment_content}
    )
    assert response.status_code == 200

    # retrieve the current total number of segments
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    data = response.json()
    original_total_items = data['data']['totalItems']

    # edit the newly added segment
    edited_content = "Edited segment content"
    segment_id_to_edit = f"{test_dataset_id}-{test_document_url}-3"
    response = client.patch(
        f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}/segment/{segment_id_to_edit}",
        json={"content": edited_content}
    )
    assert response.status_code == 200

    # verify if the segment content has been updated successfully
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    data = response.json()
    edited_segment = next((segment for segment in data['data']['segments'] if segment['segment_id'] == segment_id_to_edit), None)
    assert edited_segment is not None, "Edited segment was not found."
    assert edited_segment['content'] == edited_content, "Edited segment content does not match."

    # delete the recently edited segment
    response = client.patch(
        f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}/segment/{segment_id_to_edit}",
        json={"content": ""}
    )
    assert response.status_code == 200

    # verify if the segment has been deleted
    response = client.get(f"/v1/datasets/{test_dataset_id}/document/{test_document_uid}")
    data = response.json()
    assert data['data']['totalItems'] == original_total_items - 1, "Total items did not decrease after deletion."
    assert not any(segment['segment_id'] == segment_id_to_edit for segment in data['data']['segments']), "Deleted segment was found."

    # cleanup: delete the test dataset
    response = client.delete(f"/v1/datasets/{test_dataset_id}")
    assert response.status_code == 200
