# tpot llm

Instructions:

- `source tpot_llm/bin/activate`

Fine tuning data Format

```
{
  "messages": [
    {
      "role": "system",
      "content": "You are a tweet composer who writes engaging tweets based on given topics."
    },
    {
      "role": "user",
      "content": "Write a tweet about the following topics: achilles, archeologico nazionale, napoli, fresco briseis, pompeii, briseis, archaeohistories, agamemnon"
    },
    {
      "role": "assistant",
      "content": "Wall of Fresco Briseis taken away from Achilles; from house of tragic poet, Pompeii.\n\nAchilles sitting in his tent, Briseis is led out of tent (to Agamemnon), in presence of many Greek heroes. \n\nMuseo Archeologico Nazionale, Napoli \n\n#archaeohistories"
    }
  ]
}
```

# Modal notebooks

- `modal launch jupyter --gpu a10g`, [link](https://modal.com/docs/reference/cli/launch#modal-launch-jupyter)
