"""Incremental extraction of fictional game status tags, never spoken as dialogue."""
class ThoughtParser:
    def __init__(self):
        self.buffer = ''
        self.inside = False
        self.thought = ''

    def feed(self, text, final=False):
        self.buffer += text
        events = []
        while self.buffer:
            tag = '</thought>' if self.inside else '<thought>'
            index = self.buffer.find(tag)
            if index >= 0:
                part = self.buffer[:index]
                if self.inside:
                    self.thought += part
                    events.append(('thought', self.thought[:400]))
                    self.thought = ''
                elif part:
                    events.append(('token', part))
                self.buffer = self.buffer[index+len(tag):]
                self.inside = not self.inside
                continue
            keep = max((n for n in range(1,len(tag)) if self.buffer.endswith(tag[:n])), default=0)
            if final and keep and not self.inside:
                self.buffer = self.buffer[:-keep]
                keep = 0
            part = self.buffer[:-keep] if keep else self.buffer
            self.buffer = self.buffer[-keep:] if keep else ''
            if self.inside:
                self.thought = (self.thought + part)[:400]
            elif part:
                events.append(('token', part))
            break
        if final and self.inside:
            events.append(('thought', self.thought[:400]))
            self.thought = ''
        return events
