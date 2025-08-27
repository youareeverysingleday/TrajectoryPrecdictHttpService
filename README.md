# TrajectoryPrecdictHttpService
The trajectory prediction external service interface implemented using FastAPI.

## operation

uvicorn HttpService:app --host 0.0.0.0 --port 8000 
uvicorn HttpService:app --host 0.0.0.0 --port 8000 --reload

## input format

```json
{
    "clientID": 1,
    "userID": 2,
    "length": 100,
    "regionID": [1002, 2345, 5678, ......],
    "timestamp": ["2025-01-28 08:00:00", "2025-02-10 09:10:00", "2025-03-05 08:20:00", ......]
}
```

1. clientID 是用户身份标识，可以为空。
2. userID 是被预测对象的ID，是一个正整数值。不能为空。
3. regionID 是一个以正整数（不包含0的自然数）为元素的链表。0 是作为填充值存在的。不能为空。
4. length 表示的是 regionID 和 timestamp 所对应的链表的长度。默认情况下 length 的长度小于等于 100 。大于 100 的情况会进行截断。不能为空。
5. regionID 和 timestamp 的长度需要完全相同，而且长度等于 length 。要求越靠近链表尾部时间越晚；也就是说按照时间的降序排序。
6. timestamp 是一个以格式是 '%Y-%m-%d %H:%M:%S'的时间为元素的链表。不能为空。

### test data

```json
{
    "clientID": 100,
    "userID": 200,
    "length": 3,
    "regionID": [1002, 2345, 5678],
    "timestamp": ["2025-01-28 08:00:00", "2025-02-10 09:10:00", "2025-03-05 08:20:00"]
}
```

