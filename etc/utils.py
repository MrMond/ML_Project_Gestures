"""general utility for the project"""

from time import time, sleep


class FPS:
    """A class to be used with the ```with FPS() as fps:``` syntax to limit the framerate of a loop.
        \n**Usage**: Call ```fps.tick()``` at the end of the iteration
        \n**Attributes** (optional): limit:int
        \n**Properties** (read only): ```timestamp_ms:int```, ```measured_fps:int```"""

    def __init__(self, **kwargs):
        """A class to be used with the ```with FPS() as fps:``` syntax to limit the framerate of a loop.
        \n**Usage**: Call ```fps.tick()``` at the end of the iteration
        \n**Attributes** (optional): limit:int
        \n**Properties** (read only): ```timestamp_ms:int```, ```measured_fps:int```"""
        self.__measured_fps = 0
        self.__frame_list = []
        self.__start_time = 0
        self.__last_frame_start_time = 0
        self.__frame_limit = kwargs.get("limit")

    def __enter__(self):
        self.__start_time = time()
        self.__last_frame_start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def tick(self) -> None:
        t = time()
        self.__frame_list.append(t)
        self.__frame_list = [ts for ts in self.__frame_list if ts + 1 > t]
        self.__measured_fps = len(self.__frame_list)
        if self.__frame_limit:
            sleep_dur = 1 / self.__frame_limit - (t - self.__last_frame_start_time)
            sleep(max(0, sleep_dur))
        self.__last_frame_start_time = t

    @property
    def measured_fps(self) -> int:
        return self.__measured_fps

    @measured_fps.setter
    def measured_fps(self, x):
        pass

    @property
    def timestamp_ms(self) -> int:
        return int((time() - self.__start_time) * 100)

    @timestamp_ms.setter
    def timestamp_ms(self, x):
        pass
