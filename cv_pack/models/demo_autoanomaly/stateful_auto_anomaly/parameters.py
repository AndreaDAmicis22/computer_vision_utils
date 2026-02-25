from dataclasses import dataclass


@dataclass
class DynamicParameters:
    area_threshold: int
    anomaly_threshold: float
    disc_threshold: float


# class RedisKeyWatcher:
#     def __init__(
#         self,
#         redis_host="localhost",
#         redis_port=6379,
#         db=0,
#         redis_keys: list = [],
#         callback=None,
#     ):
#         self.redis = redis.Redis(
#             host=redis_host, port=redis_port, db=db, decode_responses=True
#         )

#         self.pubsub = self.redis.pubsub()
#         self.thread = None
#         self.keys = redis_keys
#         self.callback = callback

#     def subscribe_to_keys(self):
#         # Sottoscrive a tutte le modifiche sulle chiavi del DB
#         # "__keyspace@0__:*" ascolta il DB 0
#         for key in self.keys:
#             self.pubsub.psubscribe(**{f"__keyspace@0__:{key}": self.handle_event})
#         self.thread = self.pubsub.run_in_thread(sleep_time=0.001)

#     def handle_event(self, message):
#         if message["type"] == "pmessage":
#             key = message["channel"].split(":", 1)[1]
#             event = message["data"]
#             if self.callback:
#                 self.callback(key, str(self.redis.get(key)))
#                 # print(f"Chiave modificata: {key}, Evento: {event}")

#     def stop(self):
#         if self.thread:
#             self.thread.stop()
#             self.thread = None
#             self.pubsub.close()
