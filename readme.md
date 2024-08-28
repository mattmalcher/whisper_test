# Whisper Test

Can I use whisper to transcribe multi-channel audio?

Yes - it kind of works but:

* timestamps can be a bit out, including getting statements in the wrong order.
* repeats short phrases
    * I think this is an artifact of splitting the channels up - you generate a load of silences, which the model doesnt handle well (hallucinates.)

Might be able to do better, but could be good enough for what I need already since accuracy of recognising words is very good.

# Data Source

Took an example call from talkbank
https://sla.talkbank.org/TBB/ca/CallHome/eng/4074.cha

Saved the mp3 as wav for ease of reading in.

# Spacy model install

```py
python -m spacy download en_core_web_sm
```
