#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from bisect import bisect_left


def seg_name2id(name):
    result = 1
    levels = 'gbry'
    name = name.lower()
    idx_a = levels.find(name[0])
    if idx_a == -1:
        return result
    pos = int(name[1])
    result = 2 ** ((idx_a+1) + (pos-1) * len(levels) - 1)
    return result


class Interval:
    def __init__(self, start_time=0, end_time=0, text=''):
        if start_time < 0:
            raise ValueError(f'The time position cannot be less than 0: {start_time}')
        if end_time < 0:
            raise ValueError(f'The time position cannot be less than 0: {end_time}')
        self._text = text
        self._start_time = start_time
        self._end_time = end_time
        if self._start_time > self._end_time:
            self._start_time, self._end_time = self._end_time, self._start_time

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, other):
        if other < 0:
            raise ValueError(f'The time position cannot be less than 0: {other}')
        assert other < self._end_time, f'Start time {other} cannot be larger than an end time {self._end_time}'
        self._start_time = other

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, other):
        if other < 0:
            raise ValueError(f'The time position cannot be less than 0: {other}')
        assert self._start_time < other, f'Start time {self._start_time} cannot be larger than an end time {other}'
        self._start_time = other

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, other):
        self.text = other

    def __repr__(self):
        return f'Interval({self.start_time}, {self.end_time}, {self.text})'

    def __contains__(self, other):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time'):
            return self.start_time <= other.start_time and other.end_time <= self.end_time
        else:
            return self.start_time <= other <= self.end_time

    def __lt__(self, other):
        if hasattr(other, 'start_time'):
            return self.end_time <= other.start_time  # (1,3) is less then (3,4)
        else:
            return self.end_time <= other

    def __gt__(self, other):
        if hasattr(other, 'end_time'):
            return other.end_time <= self.start_time
        else:
            return other < self.start_time

    def __le__(self, other):
        if hasattr(other, 'end_time'):
            return self.end_time <= other.end_time
        else:
            return self.end_time <= other

    def __ge__(self, other):
        if hasattr(other, 'start_time'):
            return other.start_time <= self.start_time
        else:
            return other <= self.start_time

    def __eq__(self, other):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time'):
            if self.start_time == other.start_time and self.end_time == other.end_time:
                return True
        return False

    def __iadd__(self, other):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time') and hasattr(other, 'text'):
            self._start_time = min(self.start_time, other.start_time)
            self._end_time = max(self.end_time, other.end_time)
            self.text = self.text + other.text
        else:
            self._start_time += other
            self._end_time += other
        return self

    def __isub__(self, other):
        self._start_time -= other
        self._end_time -= other
        return self

    def duration(self):
        return self.end_time - self.start_time

    def overlaps(self, other):
        return other.start_time < self.end_time and self.start_time < other.end_time

    def get_overlap(self, other):
        if not self.overlaps(other):
            return None
        start_time = max(other.start_time, self.start_time)
        end_time = min(other.end_time, self.end_time)
        return Interval(start_time, end_time)

    def bounds(self):
        return self.start_time, self.end_time

    def to_dict(self):
        return {'text': self.text, 'min': self.start_time, 'max': self.end_time}


class Point(Interval):
    def __init__(self, time, text=''):
        super(Point, self).__init__(time, time, text)

    @property
    def time(self):
        return self.start_time

    @time.setter
    def time(self, other):
        self._start_time = other
        self._end_time = other

    def __repr__(self):
        text = self.text if self.text else None
        return f'Point({self.time}, {text})'

    def __lt__(self, other):
        if hasattr(other, 'start_time'):
            return self.time < other.start_time
        else:
            return self.time < other

    def __gt__(self, other):
        if hasattr(other, 'end_time'):
            return other.end_time < self.time
        else:
            return other < self.time

    def time2samples(self, samplerate):
        self.time = round(self.time * samplerate)


class Tier:
    def __init__(self, name='', start_time=0., end_time=0.):
        self._name = name
        self._start_time = start_time
        self._end_time = end_time
        self._objects = []

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, other):
        assert other < self._end_time, f'Start time {other} cannot be larger than an end time {self._end_time}'
        self._start_time = other

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, other):
        assert self._start_time < other, f'Start time {self._start_time} cannot be larger than an end time {other}'
        self._start_time = other

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, other):
        self.name = other

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self._objects)

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, i):
        return self._objects[i]


class PointTier(Tier):
    def __init__(self, name=None, start_time=0., end_time=0.):
        super(PointTier, self).__init__(name, start_time, end_time)

    def __str__(self):
        return f'<PointTier {self.name}, {len(self)} points>'

    def __repr__(self):
        return str(self)

    def add(self, time, text='', overwrite=False):
        self.add_point(Point(time, text), overwrite=overwrite)

    def add_point(self, point, overwrite=False):
        self.start_time = min(self.start_time, point.time)
        self.end_time = max(self.end_time, point.time)
        i = bisect_left(self._objects, point)
        if i < len(self._objects) and self._objects[i].time == point.time:
            if overwrite:
                self.remove_point(self[i])
            else:
                return
        self._objects.insert(i, point)

    def remove(self, time, text):
        self.remove_point(Point(time, text))

    def remove_point(self, point):
        self._objects.remove(point)

    def time2samples(self, samplerate):
        for pt in self._objects:
            pt.time2samples(samplerate)

    def read(self, file, encoding='utf-8'):
        """
        Read the Tier contained in the Praat PointTier
        """
        with open(file, 'r', encoding=encoding) as source:
            for i in range(3):
                source.readline()
            self.start_time = float(source.readline().strip().split()[-1])
            self.end_time = float(source.readline().strip().split()[-1])
            tier_size = int(source.readline().strip().split()[-1])
            for i in range(tier_size):
                source.readline()
                time = float(source.readline().strip().split()[-1])
                text = source.readline().strip('\n\r').split()[-1]
                self.add(time, text)

    def read_from_seg(self, file, encoding='cp1251'):
        """
        Read the Tier contained in the Seg-formatted file
        """
        with open(file, 'r', encoding=encoding) as source:
            source.readline()  # [PARAMETERS]
            samplerate = float(source.readline().split('=')[1])  # SAMPLING_FREQ
            byterate = float(source.readline().split('=')[1])  # BYTE_PER_SAMPLE
            source.readline()  # CODE
            source.readline()  # N_CHANNEL
            source.readline()  # N_LABEL
            source.readline()  # [LABELS]
            tier_name = ''
            for line in source:
                line = line.strip()
                parts = line.split(',')
                parts[2] = ','.join(parts[2:])
                time = float(parts[0])/byterate/samplerate
                tier_name = parts[1]
                text = parts[2]
                if text is None:
                    text = ''
                self.add(time, text)
            if self.name == '':
                self.name = tier_name

    def write(self, file, encoding='utf-8'):
        with open(file, 'w', encoding=encoding) as fout:
            fout.write('File type = "ooTextFile"\n')
            fout.write('Object class = "TextTier"\n\n')
            fout.write(f'xmin = {self.start_time}\n')
            fout.write(f'xmax = {self.end_time}\n')
            fout.write(f'points: size = {len(self)}\n')
            for (i, point) in enumerate(self._objects, 1):
                fout.write(f'points [{i}]:\n')
                fout.write(f'\ttime = {point.time}\n')
                fout.write(f'\ttext = {point.text}\n')

    def write_seg(self, file, tier_type=None, samplerate=22050, byterate=2, encoding='cp1251'):
        with open(file, 'w', encoding=encoding) as fout:
            level_name = tier_type
            if level_name is not None:
                level_name = seg_name2id(tier_type)
            fout.write('[PARAMETERS]\n')
            fout.write(f'SAMPLING_FREQ={samplerate}\n')
            fout.write(f'BYTE_PER_SAMPLE={byterate}\n')
            fout.write('CODE=0\n')
            fout.write('N_CHANNEL=1\n')
            fout.write(f'N_LABEL={len(self)}\n')
            fout.write('[LABELS]\n')
            for point in self._objects:
                value = round(point.time*samplerate*byterate)
                fout.write(f'{value},{level_name},{point.text}')

    def bounds(self):
        return self.start_time, self.end_time
    
    # alternative constructors
    @classmethod
    def from_file(cls, file, name=''):
        pt = cls(name=name)
        pt.read(file)
        return pt

    @classmethod
    def from_seg_file(cls, file, name=''):
        pt = cls(name=name)
        pt.read_from_seg(file)
        return pt

    @classmethod
    def from_interval_tier(cls, tier, name=''):
        pt = cls(name=name)
        if pt.name == '':
            pt.name = tier.name
        for interval in tier:
            pt.add(interval.start_time, interval.text, overwrite=True)
            pt.add(interval.end_time, '', overwrite=True)
        return pt


class IntervalTier(Tier):
    def __init__(self, name=None, start_time=0., end_time=None):
        super(IntervalTier, self).__init__(name, start_time, end_time)

    def __str__(self):
        return f'<IntervalTier {self.name}, {len(self)} intervals>'

    def __repr__(self):
        return str(self)

    def add(self, start_time, end_time, text, overwrite=False):
        self.add_interval(Interval(start_time, end_time, text), overwrite=overwrite)

    def add_interval(self, interval, overwrite=False):
        self.start_time = min(self.start_time, interval.start_time)
        self.end_time = max(self.end_time, interval.end_time)
        i = bisect_left(self._objects, interval)
        if i != len(self._objects) and self[i] == interval:
            if overwrite:
                self.remove_interval(self[i])
            return
        self._objects.insert(i, interval)

    def remove(self, start_time, end_time, text):
        self.remove_interval(Interval(start_time, end_time, text))

    def remove_interval(self, interval):
        self._objects.remove(interval)

    def read(self, file, encoding='utf-8'):
        """
        Read the Tier contained in the Praat IntervalTier
        """
        with open(file, 'r', encoding=encoding) as source:
            for i in range(3):
                source.readline()
            self.start_time = float(source.readline().strip().split()[-1])
            self.end_time = float(source.readline().strip().split()[-1])
            tier_size = int(source.readline().strip().split()[-1])
            for i in range(tier_size):
                source.readline()
                start_time = float(source.readline().strip().split()[-1])
                end_time = float(source.readline().strip().split()[-1])
                text = source.readline().strip('\n\r').split()[-1]
                self.add(start_time, end_time, text)

    def write(self, file, encoding='utf-8'):
        with open(file, 'w', encoding=encoding) as fout:
            fout.write('File type = "ooTextFile"\n')
            fout.write('Object class = "IntervalTier"\n\n')
            fout.write(f'xmin = {self.start_time}\n')
            fout.write(f'xmax = {self.end_time}\n')
            output = self._fill_in_the_gaps()
            fout.write(f'intervals: size = {len(output)}\n')
            for (i, interval) in enumerate(output, 1):
                fout.write(f'intervals [{i}]:\n')
                fout.write(f'\txmin = {interval.start_time}\n')
                fout.write(f'\txmax = {interval.end_time}\n')
                fout.write(f'\ttext = {interval.text}\n')

    def write_seg(self, file, tier_type=None, samplerate=22050, byterate=2, encoding='cp1251'):
        pt = PointTier.from_interval_tier(self)
        pt.write_seg(file, tier_type=tier_type, samplerate=samplerate, byterate=byterate, encoding=encoding)

    def bounds(self):
        return self.start_time, self.end_time

    def to_dict(self):
        return [u.to_dict() for u in self]

    def _fill_in_the_gaps(self):
        prev_t = self._start_time
        output = []
        for interval in self:
            if prev_t < interval.start_time:
                output.append(Interval(prev_t, interval.start_time))
            output.append(interval)
            prev_t = interval.end_time
        return output

    # alternative constructors
    @classmethod
    def from_file(cls, f, name=None):
        it = cls(name=name)
        it.intervals = []
        it.read(f)
        return it

    @classmethod
    def from_seg_file(cls, file, name=''):
        pt = PointTier(name=name)
        pt.read_from_seg(file)
        it = IntervalTier.from_point_tier(pt)
        return it

    @classmethod
    def from_points(cls, points, name=''):
        it = cls(name=name)
        for i in range(1, len(points)):
            it.add(points[i-1].time, points[i].time, points[i-1].text)
        return it

    @classmethod
    def from_point_tier(cls, tier, name=''):
        it = cls(name=name)
        if it.name == '':
            it.name = tier.name
        for i in range(1, len(tier)):
            it.add(tier[i-1].time, tier[i].time, tier[i-1].text)
        return it


class TextGrid(Tier):
    def __init__(self, name='', start_time=0., end_time=0.):
        super(TextGrid, self).__init__(name, start_time, end_time)

    def __str__(self):
        return f'<TextGrid {self.name}, {len(self)} Tiers>'

    def __repr__(self):
        return f'TextGrid({self.name}, {self._objects})'

    def __contains__(self, value):
        for t in self:
            if t.name == value:
                return True
        return False

    def __getitem__(self, i):
        if type(i) is str:
            result = None
            for t in self._objects:
                if t.name == i:
                    return t
        else:
            result = self._objects[i]
        return result

    def get_tier_names(self):
        return [tier.name for tier in self]

    def append(self, tier):
        self.start_time = min(self.start_time, tier.start_time)
        self.end_time = max(self.end_time, tier.end_time)
        self._objects.append(tier)

    def extend(self, tiers):
        for tier in tiers:
            self.append(tier)

    @staticmethod
    def read_file(file, encoding='utf-8'):
        with open(file, 'r', encoding=encoding) as source:
            for i in range(3):
                source.readline()
            self = TextGrid()
            self.start_time = float(source.readline().split()[-1])
            self.end_time = float(source.readline().split()[-1])
            source.readline()
            n_tiers = int(source.readline().strip().split()[-1])
            source.readline()
            for i_tier in range(n_tiers):
                source.readline()
                if source.readline().strip().split()[2] == '"IntervalTier"':
                    tier_name = source.readline().rstrip().split(' = ')[1].strip('"')
                    tier = IntervalTier(tier_name)
                    for i in range(2):
                        source.readline()
                    tier_size = int(source.readline().strip().split()[-1])
                    for j in range(tier_size):
                        source.readline()
                        start_time = float(source.readline().strip().split()[-1])
                        end_time = float(source.readline().strip().split()[-1])
                        text = source.readline().strip('\n\r').split()[-1][1:-1]
                        tier.add(start_time, end_time, text)
                    self.append(tier)
                else:  # pointTier
                    tier_name = source.readline().rstrip().split(' = ')[1].strip('"')
                    tier = PointTier(tier_name)
                    for i in range(2):
                        source.readline()
                    tier_size = int(source.readline().strip().split()[-1])
                    for j in range(tier_size):
                        source.readline()
                        time = float(source.readline().rstrip().split()[-1])
                        name = source.readline().strip('\n\r').split()[-1][1:-1]
                        tier.add(time, name)
                    self.append(tier)
        return self

    def write(self, file, encoding='utf-8'):
        with open(file, 'w', encoding=encoding) as fout:
            fout.write('File type = "ooTextFile"\n')
            fout.write('Object class = "TextGrid"\n\n')
            fout.write(f'xmin = {self.start_time}\n')
            fout.write(f'xmax = {self.end_time}\n')
            fout.write('tiers? <exists>\n')
            fout.write(f'size = {len(self)}\n')
            fout.write('item []:\n')
            for (i, tier) in enumerate(self, 1):
                fout.write(f'\titem [{i}]:\n')
                if tier.__class__ == IntervalTier:
                    fout.write('\t\tclass = "IntervalTier"\n')
                    fout.write(f'\t\tname = "{tier.name}"\n')
                    fout.write(f'\t\txmin = {tier.start_time}\n')
                    fout.write(f'\t\txmax = {tier.end_time}\n')
                    output = tier._fill_in_the_gaps()
                    fout.write(f'\t\tintervals: size = {len(output)}\n')
                    for (j, interval) in enumerate(output, 1):
                        fout.write(f'\t\t\tintervals [{j}]:\n')
                        fout.write(f'\t\t\t\txmin = {interval.start_time}\n')
                        fout.write(f'\t\t\t\txmax = {interval.end_time}\n')
                        fout.write(f'\t\t\t\ttext = "{interval.text}\n"')
                elif tier.__class__ == PointTier:
                    fout.write('\t\tclass = "TextTier"\n')
                    fout.write(f'\t\tname = "{tier.name}"\n')
                    fout.write(f'\t\txmin = {tier.start_time}\n')
                    fout.write(f'\t\txmax = {tier.end_time}\n')
                    fout.write(f'\t\tpoints: size = {len(tier)}\n')
                    for (j, point) in enumerate(tier, 1):
                        fout.write(f'\t\t\tpoints [{j}]:\n')
                        fout.write(f'\t\t\t\ttime = {point.time}\n')
                        fout.write(f'\t\t\t\ttext = "{point.text}"\n')

    @classmethod
    def from_file(cls, f, name=''):
        tg = cls(name=name)
        tg.read_file(f)
        return tg
