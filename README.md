# RetriCD

RetriCD is a compact, separate line from ConceptSkillCDM's CRG/LCRF code. It
tests whether cognitive diagnosis predictions can be supported by a student's
own query-before history instead of a global concept graph or student-id
shortcut.

The first runnable target is the text-free `full` model. Build cold-handled
chronological splits before training:

```bash
python tools/build_retricd_splits.py --source-root data_source --output-root data_retricd --nips34-match-junyi
python -m retricd.train --dataset assist_09 --data-root data_retricd --output-dir outputs
```

Server runs should use the existing ConceptSkillCDM data copied from
`~/ConceptSkillCDM/data` and the existing `xph_env` interpreter.
