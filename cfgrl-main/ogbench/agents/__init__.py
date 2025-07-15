from agents.cfgrl import CFGRLAgent
from agents.gcbc import GCBCAgent
from agents.gcfbc import GCFBCAgent
from agents.hcfgrl import HCFGRLAgent
from agents.hgcbc import HGCBCAgent
from agents.hgcfbc import HGCFBCAgent

agents = dict(
    cfgrl=CFGRLAgent,
    gcbc=GCBCAgent,
    gcfbc=GCFBCAgent,
    hcfgrl=HCFGRLAgent,
    hgcbc=HGCBCAgent,
    hgcfbc=HGCFBCAgent,
)
