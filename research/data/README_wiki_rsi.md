# Wikipedia RSI 公开支持句来源

evidence_wiki_rsi_v2.json.gz 通过 work3.wiki_rsi_references 从 evidence_dev_v1.json.gz 和 HotpotQA 官方的 hotpot_dev_fullwiki_v1.json、hotpot_dev_distractor_v1.json 生成。按官方 ID 对齐，再将原标注 title 和 sent_id 对应的句子规范化。补回 5,315 道 HotpotQA 的缺句，保留其问题键、答案哈希、标题与原有其余元数据。2WikiMultiHopQA 子包原样复用 v1。

HotpotQA 来源为 Zhilin Yang 等人的 [HotpotQA 官方数据和说明](https://github.com/hotpotqa/hotpot#data-download-and-preprocessing)。v2 中的 HotpotQA 支持句改编自该数据，按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 提供。修改为从官方配套上下文补回已有标注句，并保留规范化文本、题目散列和必要的统计。整个包只用于诊断与评估，W0 的补充段落从 P0 原检索缓存定位。

~~~bash
python -m work3.wiki_rsi_references \
  --fullwiki /path/to/hotpot_dev_fullwiki_v1.json \
  --distractor /path/to/hotpot_dev_distractor_v1.json \
  --out /tmp/evidence_wiki_rsi_v2.json.gz
~~~

生成器核验三个固定输入的 SHA-256、两个官方 dev 的题目 ID 和原标注坐标、v1 中完整记录与所有已有非空支持句。v1 保留给已冻结的 P0 轮次比较和 W0 初次诊断。
