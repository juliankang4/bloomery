//! `nodes_scale` (`docs/contrib/dgx-spark.md` §9.9): the node plans on
//! synthetic DGX Sparks, N = 1 to 4, against the cells of §9.8's "where each
//! fires" table as the predictor derives them; a clause whose cells differ
//! from the table names the difference. Three synthetic shapes carry §9.7's
//! model terms for V4.1, GLM-5.3-Flash and Qwen3.8, A low and high; no file
//! is read but where a clause reads the Qwen3.8 file's tensor table.

use std::sync::OnceLock;

use gguf::GgmlType;

use super::nodes::{
    Fabric, Fills, FillsApart, IdleWhy, LinkCost, NicBound, NodeBudget, NodeIdle, NodePlan, Nodes,
    NodesError, POST_NS, PoolOver, Terms, Workload, plan_nodes, routed_only,
};
use super::workstation::{GB10_NIC, GB10_POOL, GB10_READ, gb10, plan_on, spark_nodes};
use super::{
    Device, KvBytes, Machine, ModelTensor, ModelTensors, PlacementError, Plan, PlanLevers,
    PoolNeed, Role, Violation, plan_routed, plan_with, row_table_tier,
};
use crate::accepts::{Program, routed_rule};

/// The head's rows.
const VOCAB: u64 = 151_552;

/// The switch's hop at N = 4 [assumed, §9.7].
const HOP_NS: u64 = 1_500;

/// λ's band [derived, §9.7].
const LAMBDAS: [u64; 2] = [25_000, 70_000];

/// One prompt rate a node, S₁ (tokens a second), at each end of §9.8's pins.
const PROMPT_RATES: [u64; 2] = [1_000, 3_000];

/// Node 0's cache bytes in every cell, over its layers.
const KV: u64 = 2_000_000_000;

/// The context every plan is made at.
const CTX: u64 = 4096;

/// Picoseconds a second.
const PS: u128 = 1_000_000_000_000;

/// One synthetic shape of §9.7's model terms table: the band ends of A and
/// Sh, and the bytes beside them node 0's pool holds (`embedding`, the token
/// embedding's file bytes). Every tensor is Q4_K rows of `n_embd` values, so
/// each term is rounded down to whole rows.
#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    layers: usize,
    n_embd: u64,
    experts: u64,
    used: u64,
    expert: u64,
    shared: [u64; 2],
    serial: [u64; 2],
    embedding: u64,
}

const V41: Shape = Shape {
    name: "V4.1",
    layers: 40,
    n_embd: 5120,
    experts: 384,
    used: 6,
    expert: 16_770_000,
    shared: [800_000_000, 1_500_000_000],
    serial: [3_200_000_000, 5_800_000_000],
    embedding: 4_500_000_000,
};

/// GLM's A and Sh are the file's header read, one value each, so its A low
/// and A high cases are the same plan.
const GLM: Shape = Shape {
    name: "GLM",
    layers: 42,
    n_embd: 4096,
    experts: 288,
    used: 8,
    expert: 15_200_000,
    shared: [1_600_000_000, 1_600_000_000],
    serial: [7_360_000_000, 7_360_000_000],
    embedding: 11_000_000_000,
};

const QWEN38: Shape = Shape {
    name: "Qwen3.8",
    layers: 48,
    n_embd: 2560,
    experts: 512,
    used: 10,
    expert: 3_170_000,
    shared: [140_000_000, 140_000_000],
    serial: [3_400_000_000, 4_400_000_000],
    embedding: 1_000_000_000,
};

const SHAPES: [Shape; 3] = [V41, GLM, QWEN38];

impl Shape {
    /// One Q4_K row's bytes.
    fn row(self) -> u64 {
        self.n_embd / 256 * 144
    }

    /// Rows a layer at A low (`high` false) or high: attention, the shared
    /// expert, and one expert.
    fn rows(self, high: bool) -> (u64, u64, u64) {
        let at = usize::from(high);
        let (row, layers) = (self.row(), self.layers as u64);
        (
            (self.serial[at] / row - VOCAB) / layers,
            self.shared[at] / row / layers,
            self.expert / row,
        )
    }

    /// The bytes a decode token reads of [`Shape::model`] on one node: A
    /// (attention, the head and one embedding row), and a layer's Sh and one
    /// expert.
    fn bytes(self, high: bool) -> (u64, u64, u64) {
        let (attn, shared, expert) = self.rows(high);
        let row = self.row();
        (
            (attn * self.layers as u64 + VOCAB + 1) * row,
            shared * row,
            expert * row,
        )
    }

    /// The shape as a model's tensors at A low (`high` false) or high: per
    /// layer one attention tensor and one shared expert, and one routed stack
    /// of `experts` experts of `expert` bytes; a head and a token embedding.
    /// A's attention part is A less the head.
    fn model(self, high: bool) -> ModelTensors {
        let row = self.row();
        let (attn, shared, expert) = self.rows(high);
        let tensor = |name: String, layer, role, rows: Vec<u64>| {
            let file_bytes = row * rows.iter().product::<u64>();
            ModelTensor {
                name,
                shard: 0,
                layer,
                role,
                ty: GgmlType::Q4_K,
                dims: [self.n_embd].into_iter().chain(rows).collect(),
                file_bytes,
                gathered_rows: (role == Role::TokenEmbedding).then_some(1),
            }
        };
        let mut tensors = vec![tensor(
            "embedding".into(),
            None,
            Role::TokenEmbedding,
            vec![self.embedding / row],
        )];
        for l in 0..self.layers {
            tensors.push(tensor(
                format!("attn {l}"),
                Some(l),
                Role::Attention,
                vec![attn],
            ));
            tensors.push(tensor(
                format!("shexp {l}"),
                Some(l),
                Role::SharedExpert,
                vec![shared],
            ));
            tensors.push(tensor(
                format!("experts {l}"),
                Some(l),
                Role::RoutedExperts,
                vec![expert, self.experts],
            ));
        }
        tensors.push(tensor("head".into(), None, Role::Head, vec![VOCAB]));
        ModelTensors {
            tensors,
            layers: self.layers,
            experts: self.experts,
            experts_used: self.used,
        }
    }
}

/// §9.9's fabric for `n` nodes at round trip `lambda_ns`: one cable, a ring
/// of three, a switch for four.
fn fabric(n: usize, lambda_ns: u64) -> Fabric {
    let link = LinkCost {
        round_trip_ns: lambda_ns,
    };
    match n {
        1 => Fabric::Alone,
        2 => Fabric::Pair { link },
        3 => Fabric::Ring { link },
        _ => Fabric::Switch {
            link,
            hop_ns: HOP_NS,
        },
    }
}

fn workload(prompt_rate: u64) -> Workload {
    Workload {
        prompt_rate,
        whole: false,
        fills: None,
    }
}

/// The same cache bytes on every layer.
struct Kv(u64);

impl KvBytes for Kv {
    fn layer_bytes(&self, _: usize, _: u64) -> u64 {
        self.0
    }
}

/// Today's card plan of `model` on `machine`, one node: node 0's with
/// [`KV`] over the layers, a remote's ([`routed_only`]) with none.
fn today<'a>(model: &'a ModelTensors, machine: &'a Machine, kv: bool) -> Plan<'a> {
    let kv = Kv(if kv { KV / model.layers as u64 } else { 0 });
    plan_with(model, machine, CTX, &kv, None).expect("today's plan")
}

/// A Spark of `free` bytes for `layers`.
fn spark(free: u64, layers: usize) -> Machine {
    plan_on(gb10(free), layers)
}

/// `model`'s node budgets: node 0's on `machines[0]`, each other node's of
/// its routed stacks on its own machine; the nodes of `nodes` take the
/// machines' nodes.
fn budgets_on(model: &ModelTensors, nodes: &mut Nodes, machines: &[Machine]) -> Vec<NodeBudget> {
    let remote = routed_only(model);
    let mut budgets = Vec::new();
    for (j, m) in machines.iter().enumerate() {
        nodes.machine.nodes[j] = m.nodes[0].clone();
        let plan = if j == 0 {
            today(model, m, true)
        } else {
            today(&remote, m, false)
        };
        budgets.push(NodeBudget::of(&plan).expect("a budget"));
    }
    budgets
}

/// The terms of `model` on `machine` as node 0's plan reads them.
fn terms_of(model: &ModelTensors, machine: &Machine, n_embd: u64) -> Terms {
    Terms::of(&today(model, machine, true), n_embd).expect("the terms")
}

/// A shape's terms and its budgets: node 0's, a remote GB10's.
struct Case {
    shape: Shape,
    high: bool,
    terms: Terms,
    budgets: [NodeBudget; 2],
}

/// Every shape at A low and high, planned once.
fn cases() -> &'static [Case] {
    static CASES: OnceLock<Vec<Case>> = OnceLock::new();
    CASES.get_or_init(|| {
        let mut cases = Vec::new();
        for shape in SHAPES {
            let machine = spark(GB10_POOL, shape.layers);
            for high in [false, true] {
                let model = shape.model(high);
                let remote = routed_only(&model);
                let plan = today(&model, &machine, true);
                let budget = |p: &Plan<'_>| NodeBudget::of(p).expect("a budget");
                cases.push(Case {
                    shape,
                    high,
                    terms: Terms::of(&plan, shape.n_embd).expect("the terms"),
                    budgets: [budget(&plan), budget(&today(&remote, &machine, false))],
                });
            }
        }
        cases
    })
}

impl Case {
    /// The budgets of `n` Sparks.
    fn budgets(&self, n: usize) -> Vec<NodeBudget> {
        (0..n).map(|j| self.budgets[j.min(1)].clone()).collect()
    }

    /// The plan on `nodes` for `workload`.
    fn plan<'a>(
        &'a self,
        nodes: &'a Nodes,
        budgets: &'a [NodeBudget],
        workload: Workload,
    ) -> Result<NodePlan<'a>, NodesError> {
        plan_nodes(nodes, &self.terms, budgets, workload)
    }
}

/// One cell of §9.8's table: a model, N, λ (µs) and A high.
type Cell = (&'static str, usize, u64, bool);

/// Every cell's plan at `prompt_rate` on nodes of `ruler_permille`, handed to
/// `check` with its cell, N the innermost loop.
fn each_cell(ruler_permille: u64, prompt_rate: u64, mut check: impl FnMut(Cell, &NodePlan<'_>)) {
    for case in cases() {
        for lambda in LAMBDAS {
            for n in 1..=4 {
                let mut nodes = spark_nodes(n, case.shape.layers, fabric(n, lambda));
                nodes.ruler_permille = ruler_permille;
                let budgets = case.budgets(n);
                let plan = case
                    .plan(&nodes, &budgets, workload(prompt_rate))
                    .unwrap_or_else(|e| panic!("{} N = {n}: {e}", case.shape.name));
                check((case.shape.name, n, lambda / 1000, case.high), &plan);
            }
        }
    }
}

/// The case of `cell`.
fn case_of(cell: Cell) -> &'static Case {
    cases()
        .iter()
        .find(|c| c.shape.name == cell.0 && c.high == cell.3)
        .expect("a case")
}

/// The box ruler.
const RULER: u64 = super::workstation::RULER;

/// A one-layer model of `experts` ids of `bytes` each, `used` a token,
/// beside a head of `4 × bytes`: F32 rows of one value.
fn tiny(experts: u64, used: u64, bytes: u64) -> ModelTensors {
    let tensor = |name: &str, layer, role, ty, dims: Vec<u64>, file_bytes| ModelTensor {
        name: name.into(),
        shard: 0,
        layer,
        role,
        ty,
        dims,
        file_bytes,
        gathered_rows: None,
    };
    ModelTensors {
        tensors: vec![
            tensor(
                "head",
                None,
                Role::Head,
                GgmlType::F32,
                vec![1, bytes],
                4 * bytes,
            ),
            tensor(
                "experts 0",
                Some(0),
                Role::RoutedExperts,
                GgmlType::I8,
                vec![1, bytes, experts],
                experts * bytes,
            ),
        ],
        layers: 1,
        experts,
        experts_used: used,
    }
}

/// One-layer Sparks for `model`, of `id`-byte ids, whose budgets leave
/// room for `rooms` ids: node 0's beside its cache, each other's of the
/// routed stacks alone. Each pool is half an id over, since a smaller
/// pool's own budget is smaller.
fn with_room(model: &ModelTensors, id: u64, rooms: &[u64]) -> Vec<Machine> {
    let wide = spark(GB10_POOL, 1);
    let remote = routed_only(model);
    let machines: Vec<Machine> = rooms
        .iter()
        .enumerate()
        .map(|(j, &room)| {
            let (model, kv) = if j == 0 {
                (model, true)
            } else {
                (&remote, false)
            };
            let fixed = NodeBudget::of(&today(model, &wide, kv))
                .expect("a budget")
                .fixed();
            spark(fixed + room * id + id / 2, 1)
        })
        .collect();
    for (j, (m, &room)) in machines.iter().zip(rooms).enumerate() {
        let model = if j == 0 { model } else { &remote };
        let b = NodeBudget::of(&today(model, m, j == 0)).expect("a budget");
        assert_eq!((b.pool() - b.fixed()) / id, room, "node {j}");
    }
    machines
}

/// The nodes that own an id, node 0 always.
fn used(plan: &NodePlan<'_>) -> u64 {
    plan.owned()
        .iter()
        .enumerate()
        .filter(|&(j, &o)| j == 0 || o > 0)
        .count() as u64
}

/// Clause 1, `n1_is_today`: one Spark's machine is [`plan_on`] the GB10,
/// and its plan is today's: held whole, node 0's bytes are today's pool sum
/// ([`PoolNeed`]), or both refuse it by the same bytes; paged, node 0 holds
/// today's terms and every id its pool has room for. No link term, no
/// message, no NIC, no idle node, and the step and node 0's window the bytes
/// the shape lays out over the pool's read rate. The terms read the plan:
/// a BF16 head and shared expert in the card's f32, an engram table on the
/// host in its gathered rows, one on the NVMe tier refused by name, and the Qwen3.8
/// file's row width as given, whatever its head tensors' order (its model
/// tensors and first two layers, planned by its program's routed rule). A host
/// reserve on node 0 raises its bytes by itself.
#[test]
fn n1_is_today() {
    for case in cases() {
        let (shape, at) = (case.shape, (case.shape.name, case.high));
        let nodes = spark_nodes(1, shape.layers, Fabric::Alone);
        assert_eq!(nodes.machine, spark(GB10_POOL, shape.layers));
        let model = shape.model(case.high);
        let today = today(&model, &nodes.machine, true);
        let Some(PoolNeed { cards, host, pool }) = today.pool_need() else {
            panic!("{at:?}: a GB10 has a pool");
        };
        let need = cards + host;
        let broken = today.violations();
        assert!(
            broken
                .iter()
                .all(|v| matches!(v, Violation::PoolOver { .. } | Violation::HostOver { .. })),
            "{at:?}: {broken:?}"
        );
        let budgets = case.budgets(1);
        let whole = Workload {
            whole: true,
            ..workload(3_000)
        };
        let (serial, shared, expert) = shape.bytes(case.high);
        let id = expert * shape.layers as u64;
        match case.plan(&nodes, &budgets, whole) {
            Ok(plan) => {
                assert!(broken.is_empty(), "{at:?}: {broken:?}");
                assert_eq!(plan.predict().expect("a prediction").per_node[0].pool, need);
            }
            Err(NodesError::PoolOver(over)) => {
                assert!(broken.contains(&Violation::PoolOver {
                    cards,
                    host,
                    beside: 0,
                    pool
                }));
                assert_eq!(
                    over,
                    PoolOver {
                        node: 0,
                        need,
                        pool
                    },
                    "{at:?}"
                );
            }
            Err(e) => panic!("{at:?}: {e}"),
        }
        let plan = case
            .plan(&nodes, &budgets, workload(3_000))
            .expect("one Spark plans");
        let p = plan.predict().expect("a prediction");
        let held = p.per_node[0].pool;
        assert!(
            held <= pool && (need - held).is_multiple_of(id) && (need == held || pool - held < id),
            "{at:?}: {held} B held, today {need} B in {pool} B"
        );
        let window = (shared + shape.used * expert) * shape.layers as u64;
        let ns = |bytes: u64| u128::from(bytes) * 1_000_000_000 / u128::from(GB10_READ);
        assert_eq!(plan.owned(), [shape.experts], "{at:?}");
        assert!(plan.idle().is_empty(), "{at:?}: {:?}", plan.idle());
        assert_eq!(
            (
                p.link_ns,
                p.per_node[0].msgs_per_layer,
                p.per_node[0].nic_permille
            ),
            (0, 0, 0),
            "{at:?}"
        );
        assert!(
            u128::from(p.step_ns).abs_diff(ns(serial + window)) <= 1
                && u128::from(p.per_node[0].window_ns).abs_diff(ns(window)) <= 1,
            "{at:?}: step {} ns and window {} ns, the shape's {} and {} ns",
            p.step_ns,
            p.per_node[0].window_ns,
            ns(serial + window),
            ns(window)
        );
    }
    let step_of = |plan: &Plan<'_>, n_embd| {
        let nodes = spark_nodes(1, plan.model.layers, Fabric::Alone);
        let terms = Terms::of(plan, n_embd).expect("the terms");
        let budgets = [NodeBudget::of(plan).expect("a budget")];
        let step = plan_nodes(&nodes, &terms, &budgets, workload(3_000))
            .and_then(|p| p.predict())
            .expect("one Spark")
            .step_ns;
        (terms, step)
    };
    let (shape, n) = (QWEN38, QWEN38.n_embd);
    let row = shape.row();
    let (attn, sh, _) = shape.rows(false);
    let mut model = shape.model(false);
    for t in model
        .tensors
        .iter_mut()
        .filter(|t| t.name == "head" || t.name == "shexp 0")
    {
        t.ty = GgmlType::BF16;
        t.file_bytes = t.dims.iter().product::<u64>() * 2;
    }
    model.tensors.push(ModelTensor {
        name: "engram 0".into(),
        shard: 0,
        layer: Some(0),
        role: Role::EngramTable,
        ty: GgmlType::Q4_K,
        dims: vec![n, 1000],
        file_bytes: 1000 * row,
        gathered_rows: Some(4),
    });
    let machine = spark(GB10_POOL, shape.layers);
    let mut plan = today(&model, &machine, true);
    assert_eq!(plan.row_tier().expect("one tier"), Some(Device::Nvme));
    let nvme = |plan: &Plan<'_>, n_embd| match Terms::of(plan, n_embd) {
        Err(NodesError::Model(PlacementError::Tensor { name, .. })) => name,
        terms => panic!("a table on the NVMe tier: {terms:?}"),
    };
    assert_eq!(nvme(&plan, n), "engram 0");
    row_table_tier(&mut plan, u64::MAX, 0, 1).expect("the table's tier");
    assert_eq!(plan.row_tier().expect("one tier"), Some(Device::Host));
    let (t, step) = step_of(&plan, n);
    let serial = (attn * shape.layers as u64 + 1) * row + VOCAB * n * 4;
    let shared = sh * row * (shape.layers as u64 - 1) + sh * n * 4;
    let shared_of = |t: &Terms| t.layers.iter().map(|l| l.shared).sum::<u64>();
    assert_eq!((t.serial, shared_of(&t)), (serial + 4 * row, shared));
    let bytes = t.serial + shared + shape.used * shape.layers as u64 * t.layers[0].expert;
    let ns = u128::from(bytes) * 1_000_000_000 / u128::from(GB10_READ);
    assert!(
        u128::from(step).abs_diff(ns) <= 1,
        "step {step} ns, {ns} ns"
    );
    let file = super::qwen38_cards::model(false);
    let file = ModelTensors {
        tensors: file
            .tensors
            .into_iter()
            .filter(|t| t.layer.is_none_or(|l| l < 2))
            .collect(),
        layers: 2,
        ..file
    };
    let machine = spark(GB10_POOL, file.layers);
    let kv = Kv(KV / file.layers as u64);
    let rule = routed_rule(Program::Qwen38Body);
    let mut plan = plan_routed(&file, &machine, CTX, &kv, &PlanLevers::default(), rule)
        .expect("the file's plan");
    let (i, ple) = file
        .tensors
        .iter()
        .enumerate()
        .find(|(_, t)| t.role == Role::EngramTable)
        .expect("the PLE table");
    assert_eq!(nvme(&plan, 2560), ple.name);
    row_table_tier(&mut plan, u64::MAX, 0, 1).expect("the PLE table's tier");
    let on_host = Terms::of(&plan, 2560).expect("the file's terms");
    let mut unread = plan.clone();
    for r in unread.rows.iter_mut().filter(|r| r.tensor == i) {
        r.read_bytes = 0;
    }
    let unread = Terms::of(&unread, 2560).expect("the file's terms");
    let gathered = ple.gathered_rows.expect("its rows") * ple.file_bytes
        / ple.dims[1..].iter().product::<u64>();
    assert_eq!(
        (on_host.n_embd, on_host.serial - unread.serial),
        (2560, gathered)
    );
    let model = QWEN38.model(false);
    let reserve = 4 << 30;
    let mut machine = spark(GB10_POOL, QWEN38.layers);
    let base = NodeBudget::of(&today(&model, &machine, true)).expect("a budget");
    machine
        .host_mut()
        .reserves
        .push(("checkpoints".to_string(), reserve));
    let reserved = NodeBudget::of(&today(&model, &machine, true)).expect("a budget");
    let terms = &case_of(("Qwen3.8", 1, 25, false)).terms;
    let held = |machine: Machine, budget| {
        let nodes = Nodes {
            machine,
            ..spark_nodes(1, QWEN38.layers, Fabric::Alone)
        };
        let budgets = [budget];
        let plan = plan_nodes(&nodes, terms, &budgets, workload(3_000)).expect("one Spark");
        plan.predict().expect("a prediction").per_node[0].pool
    };
    assert_eq!(
        held(machine, reserved),
        held(spark(GB10_POOL, QWEN38.layers), base) + reserve
    );
}

/// Clause 2, `pool_per_node`: each used node's planned bytes are held to
/// its own pool, a remote's beside its own card's set-aside and margin, and
/// a node the plan leaves idle holds none. V4.1 whole is refused at N = 2
/// by a node's pool and placed at N = 4; no cell of §9.8 is refused
/// (`each_cell` panics on a refusal). A node whose own budget passes its
/// pool is named idle for it, not refused, whether node 0 holds the model or
/// pages; a node is used while the pools before it page, whatever it gains,
/// unless its own pool has no room; and a node the final split gives no id is
/// named idle, also one the loop took while node 0 paged and a later node's
/// room emptied.
#[test]
fn pool_per_node() {
    each_cell(RULER, 3_000, |cell, plan| {
        let over = plan.over().expect("a plan's sums");
        assert!(over.is_empty(), "{cell:?}: {over:?}");
        let id: u64 = plan.terms().layers.iter().map(|l| l.expert).sum();
        let p = plan.predict().expect("a prediction");
        let card = &plan.nodes().machine.cards[0];
        let remote = &case_of(cell).budgets[1];
        assert!(remote.fixed() >= card.set_aside_bytes() + card.margin_bytes);
        for (j, &owned) in plan.owned().iter().enumerate().skip(1) {
            let ids = p.per_node[j].pool.checked_sub(remote.fixed());
            assert!(
                (owned == 0 && p.per_node[j].pool == 0)
                    || ids.is_some_and(|b| b > 0 && b % id == 0),
                "{cell:?}: node {j}"
            );
        }
    });
    let whole = Workload {
        whole: true,
        ..workload(3_000)
    };
    let case = &cases()[0];
    let at = |n| {
        let nodes = spark_nodes(n, V41.layers, fabric(n, LAMBDAS[0]));
        let budgets = case.budgets(n);
        case.plan(&nodes, &budgets, whole)
            .map(|p| p.owned().to_vec())
    };
    assert!(
        matches!(at(2), Err(NodesError::PoolOver(_))),
        "V4.1 whole at N = 2: {:?}",
        at(2)
    );
    let owned = at(4).expect("V4.1 whole at N = 4");
    assert_eq!(owned.iter().sum::<u64>(), V41.experts);

    let model = QWEN38.model(false);
    let mut pinned = spark(GB10_POOL, QWEN38.layers);
    pinned
        .host_mut()
        .reserves
        .push(("pinned".to_string(), 120_000_000_000));
    let mut nodes = spark_nodes(2, QWEN38.layers, fabric(2, LAMBDAS[0]));
    let machines = [spark(GB10_POOL, QWEN38.layers), pinned.clone()];
    let budgets = budgets_on(&model, &mut nodes, &machines);
    assert!(budgets[1].fixed() > budgets[1].pool());
    let plan = cases()[4]
        .plan(&nodes, &budgets, whole)
        .expect("node 0 holds Qwen3.8");
    let over = IdleWhy::Pool {
        fixed: budgets[1].fixed(),
        pool: budgets[1].pool(),
    };
    assert_eq!(plan.owned(), [QWEN38.experts, 0]);
    assert_eq!(plan.idle(), [NodeIdle { node: 1, why: over }]);
    assert_eq!(plan.predict().expect("a prediction").per_node[1].pool, 0);

    let shape = Shape {
        embedding: 40_000_000_000,
        ..QWEN38
    };
    let model = shape.model(false);
    let machine = spark(GB10_POOL, shape.layers);
    let terms = terms_of(&model, &machine, shape.n_embd);
    let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[1]));
    nodes.ruler_permille = 4 * RULER;
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("two Sparks");
    let id = shape.bytes(false).2 * shape.layers as u64;
    let rooms = |budgets: &[NodeBudget]| -> Vec<u64> {
        budgets
            .iter()
            .map(|b| b.pool().saturating_sub(b.fixed()) / id)
            .collect()
    };
    let room = rooms(&budgets);
    assert!(
        room[0] < shape.experts && plan.owned().iter().zip(&room).all(|(o, r)| o <= r),
        "{room:?}: {:?}",
        plan.owned()
    );
    let mut full = machine.clone();
    full.host_mut().reserves.push((
        "full".to_string(),
        budgets[1].pool() - budgets[1].fixed() - id / 2,
    ));
    let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[0]));
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), pinned]);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("node 0 pages");
    let over = IdleWhy::Pool {
        fixed: budgets[1].fixed(),
        pool: budgets[1].pool(),
    };
    assert_eq!(plan.owned(), [shape.experts, 0]);
    assert_eq!(plan.idle(), [NodeIdle { node: 1, why: over }]);
    let mut nodes = spark_nodes(3, shape.layers, fabric(3, LAMBDAS[0]));
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), full, machine]);
    let room = rooms(&budgets);
    assert!(room[0] < shape.experts && room[1] == 0 && budgets[1].fixed() <= budgets[1].pool());
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("three Sparks");
    assert_eq!(plan.owned()[1], 0, "{room:?}: {:?}", plan.owned());
    assert_eq!(plan.idle().iter().map(|i| i.node).collect::<Vec<_>>(), [1]);

    let shape = Shape {
        name: "two ids",
        layers: 1,
        experts: 2,
        used: 1,
        expert: 40_000_000_000,
        ..QWEN38
    };
    let model = shape.model(false);
    let mut nodes = spark_nodes(3, 1, fabric(3, LAMBDAS[0]));
    let machines: Vec<Machine> = [60_000_000_000, 20_000_000_000, 60_000_000_000]
        .into_iter()
        .map(|free| spark(free, 1))
        .collect();
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], shape.n_embd);
    let id = shape.bytes(false).2;
    let rooms: Vec<u64> = budgets
        .iter()
        .map(|b| (b.pool() - b.fixed()) / id)
        .collect();
    assert_eq!(rooms, [1, 0, 1]);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("three Sparks");
    assert_eq!(plan.owned(), [1, 0, 1]);
    assert_eq!(plan.idle().iter().map(|i| i.node).collect::<Vec<_>>(), [1]);
    let mut nodes = spark_nodes(2, 1, fabric(2, LAMBDAS[0]));
    nodes.ruler_permille = 1000;
    let budgets = budgets_on(&model, &mut nodes, &machines[..2]);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("two Sparks");
    assert_eq!(plan.owned(), [2, 0]);
    assert_eq!(plan.idle().iter().map(|i| i.node).collect::<Vec<_>>(), [1]);
}

/// Whether `plan`'s slowest window is at least the rule's optimum and
/// within (1 + 2.5 × ruler) of it, in picoseconds; the windows and the
/// optimum, for the message.
fn balanced(plan: &NodePlan<'_>) -> (bool, Vec<u128>, u128) {
    let windows: Vec<u128> = (0..plan.owned().len())
        .map(|j| plan.window_ps(j).expect("a window"))
        .collect();
    let slowest = windows.iter().copied().max().unwrap_or(0);
    let optimum = plan.balanced_window_ps().expect("the optimum");
    let held = optimum <= slowest && 2000 * slowest <= (2000 + 5 * u128::from(RULER)) * optimum;
    (held, windows, optimum)
}

/// Clause 3, `read_balance`: on every plan the slowest node's window is at
/// least the rule's optimum and within (1 + 2.5 × ruler) of it, also where
/// two pools hold more ids than a u64 counts. The windows are read in
/// picoseconds: three ids of 10⁹ B at top-k 1 read at 10¹⁸ B/s on two nodes
/// of one id's room each split [2, 1], whose windows of 666 and 333 ps are a
/// third past the optimum's 500 ps, which the check refuses.
#[test]
fn read_balance() {
    let check = |cell: Cell, plan: &NodePlan<'_>| {
        let (held, windows, optimum) = balanced(plan);
        assert!(
            held,
            "{cell:?}: windows {windows:?} ps, optimum {optimum} ps"
        );
    };
    each_cell(RULER, 3_000, check);
    let model = tiny(2, 2, 1);
    let machine = spark(3 << 62, 1);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    nodes.read = 1;
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = Terms::of(&today(&model, &machine, false), 1).expect("the terms");
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("two nodes");
    assert_eq!(plan.owned(), [1, 1]);
    check(("u64 pools", 2, 0, false), &plan);
    let id = 1_000_000_000;
    let model = tiny(3, 1, id);
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    nodes.read = 1_000_000_000_000_000_000;
    let machines = with_room(&model, id, &[1, 1]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(1)).expect("two nodes");
    assert_eq!(plan.owned(), [2, 1]);
    assert_eq!(balanced(&plan), (false, vec![666, 333], 500));
}

/// Clause 4, `link_share`: the chosen plan's link term is at most a tenth of
/// its step, and a node is dropped for it exactly where criterion 2 fires,
/// as §9.8's table names: Qwen3.8 at N = 4, λ 70, A low, which then uses
/// two of the four nodes. The share is read on the exact steps: three ids
/// of 1,000 B at top-k 1 read at 10¹¹ B/s, λ 17,000 ps, split [2, 1],
/// expose 5,666⅔ of 55,666⅔ ps (10.2 %), and three of 19,199 B at
/// 10¹² B/s, λ 32,000 ps, expose 10,666⅔ of 106,661⅔ ps, past a tenth by
/// ½ ps.
#[test]
fn link_share() {
    let mut fired = Vec::new();
    each_cell(RULER, 3_000, |cell, plan| {
        let step = plan.step_ps().expect("a step");
        let link = plan.exposed_ps(step).expect("the link");
        assert!(
            link.times(10).expect("ten links") <= step,
            "{cell:?}: link {link:?} of step {step:?}"
        );
        if plan
            .idle()
            .iter()
            .any(|i| matches!(i.why, IdleWhy::LinkShare { .. }))
        {
            fired.push(cell);
            assert_eq!(used(plan), 2, "{cell:?}");
        }
    });
    assert_eq!(fired, vec![("Qwen3.8", 4, 70, false)]);
    let model = tiny(3, 1, 1_000);
    let machine = spark(GB10_POOL, 1);
    let link = LinkCost { round_trip_ns: 17 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.ruler_permille) = (100_000_000_000, 1);
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = terms_of(&model, &machine, 1);
    let whole = Workload {
        whole: true,
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        plan.idle(),
        [NodeIdle {
            node: 1,
            why: IdleWhy::LinkShare {
                link_ns: 5,
                step_ns: 55
            }
        }]
    );
    let model = tiny(3, 1, 19_199);
    let link = LinkCost { round_trip_ns: 32 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.ruler_permille) = (1_000_000_000_000, 1);
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = terms_of(&model, &machine, 1);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        plan.idle(),
        [NodeIdle {
            node: 1,
            why: IdleWhy::LinkShare {
                link_ns: 10,
                step_ns: 106
            }
        }]
    );
}

/// Node 0's NIC, in permille, on a plan of `owned` ids: a token's mean
/// rows, one a slot on a remote, kept a fraction, against the decode step
/// and a prompt batch at S₁ times the one-node token's bytes over A and the
/// slowest node's FFN, node 0's with Sh.
fn nic_of(shape: Shape, owned: &[u64], step_ns: u64, s1: u64) -> u128 {
    let (a, sh, e) = shape.bytes(false);
    let layers = shape.layers as u64;
    let (sh, id) = (u128::from(sh * layers), u128::from(e * layers));
    let (k, ex) = (u128::from(shape.used), u128::from(shape.experts));
    let remote: u128 = owned[1..].iter().map(|&o| u128::from(o)).sum();
    let bytes = u128::from(layers * 4 * shape.n_embd) * k * remote;
    let nic = u128::from(GB10_NIC);
    let decode = bytes * PS * 1000 / (u128::from(step_ns) * 1000 * nic * ex);
    let ffn = owned
        .iter()
        .enumerate()
        .map(|(j, &o)| k * u128::from(o) * id + if j == 0 { sh * ex } else { 0 })
        .max()
        .unwrap_or(0);
    let one = u128::from(a) + sh + k * id;
    let prompt = u128::from(s1) * one * bytes * 1000 / ((u128::from(a) * ex + ffn) * nic);
    decode.max(prompt)
}

/// Clause 5, `nic_not_max`: a plan whose NIC passes half a decode step or a
/// prompt batch names the node and its busy share, which is the node's
/// reported one and the busiest, exactly where criterion 5 fires on the
/// chosen plan at its split: node 0 takes the rows of the remotes' share of
/// the slots. At S₁ 3,000 that is V4.1 and GLM at N ≥ 2 but three of
/// V4.1's four cells at N = 2 (all but λ 25, A low, on the line at 500 ‰),
/// and never Qwen3.8, whose node 0 keeps the most ids; none at 1,000.
/// §9.8's table, from an even split, names N ≥ 3 for all three. The prompt
/// counts the split's own shares: a remote that owns every id gains the
/// prompt only Sh. A token's mean bytes stay a fraction: three ids a layer
/// at top-k 1, owned [1, 2], take 8/3 B a token, two thirds of a 4 B row,
/// and node 0's prompt share is 4/7, past the line. The decode line reads
/// the exact step: the same split of ids of 666,666,664 B read at 10¹⁰ B/s,
/// a NIC of 16 B/s and λ 2,000 ps take a step of 333,333,333,333⅓ ps, twice
/// node 0's NIC time of 166,666,666,666⅔ ps, so on the line.
#[test]
fn nic_not_max() {
    for rate in PROMPT_RATES {
        let mut fired = Vec::new();
        let mut expected = Vec::new();
        each_cell(RULER, rate, |cell, plan| {
            let bound = plan.nic_bound().expect("a prediction");
            let p = plan.predict().expect("a prediction");
            let busiest = p.per_node.iter().map(|l| l.nic_permille).max().unwrap_or(0);
            match bound {
                Some(b) => {
                    fired.push(cell);
                    assert!(
                        b.busy_permille == p.per_node[b.node].nic_permille
                            && b.busy_permille == busiest
                            && busiest >= 500,
                        "{cell:?}: {b:?}, {:?}",
                        p.per_node
                    );
                }
                None => assert!(busiest <= 500, "{cell:?}: {:?}", p.per_node),
            }
            let below =
                cell.0 == "V4.1" && cell.1 == 2 && cell.2 == 70 || cell == ("V4.1", 2, 25, true);
            if rate == 3_000 && cell.1 >= 2 && cell.0 != "Qwen3.8" && !below {
                expected.push(cell);
            }
        });
        assert_eq!(fired, expected, "S₁ {rate}");
    }
    let shape = Shape {
        name: "remote FFN",
        experts: 2,
        used: 2,
        expert: 100_000_000,
        ..V41
    };
    let model = shape.model(false);
    let wide = spark(GB10_POOL, shape.layers);
    let fixed = NodeBudget::of(&today(&model, &wide, true))
        .expect("a budget")
        .fixed();
    let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[0]));
    let machines = [spark(fixed + 2_000_000_000, shape.layers), wide];
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], shape.n_embd);
    let s1 = 6_000;
    let whole = Workload {
        prompt_rate: s1,
        whole: true,
        fills: None,
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two Sparks");
    assert_eq!(plan.owned(), [0, 2]);
    let p = plan.predict().expect("a prediction");
    let busy = nic_of(shape, plan.owned(), p.step_ns, s1);
    assert!(
        u128::from(p.per_node[0].nic_permille).abs_diff(busy) <= 1,
        "{} ‰, {busy} ‰",
        p.per_node[0].nic_permille
    );
    assert_eq!(plan.nic_bound().expect("a prediction"), None);
    let id = 1_000_000_000;
    let model = tiny(3, 1, id);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.nic, nodes.ruler_permille) = (1, 5, 1);
    let machines = with_room(&model, id, &[1, 2]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let whole = Workload {
        prompt_rate: 1,
        whole: true,
        fills: None,
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(plan.owned(), [1, 2]);
    let p = plan.predict().expect("a prediction");
    assert_eq!(
        (
            plan.nic_bound().expect("a prediction"),
            p.per_node[0].nic_permille
        ),
        (
            Some(NicBound {
                node: 0,
                busy_permille: 571
            }),
            571
        )
    );
    let id = 666_666_664;
    let model = tiny(3, 1, id);
    let link = LinkCost { round_trip_ns: 2 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.nic, nodes.ruler_permille) = (10_000_000_000, 16, 1);
    let machines = with_room(&model, id, &[1, 2]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(plan.owned(), [1, 2]);
    let p = plan.predict().expect("a prediction");
    assert_eq!(
        (
            plan.nic_bound().expect("a prediction"),
            p.per_node[0].nic_permille
        ),
        (None, 500)
    );
}

/// Clause 6, `messages_o_n`: per layer the busiest node posts
/// 2 × min(k, N − 1) messages over the N nodes the plan uses, never one a
/// slot, and their posts take at most 2 % of the layer. Also where k is
/// under N − 1: one expert a token of seven, two ids a pool, on four Sparks
/// the pools all need, where a token's handoff goes only to the node its
/// slot lands on, and the step and node 0's NIC are that rule's.
#[test]
fn messages_o_n() {
    let check = |cell: Cell, plan: &NodePlan<'_>| {
        let p = plan.predict().expect("a prediction");
        let busiest = p
            .per_node
            .iter()
            .map(|l| l.msgs_per_layer)
            .max()
            .unwrap_or(0);
        let k = plan.terms().experts_used;
        let layers = plan.terms().layers.len() as u64;
        assert_eq!(busiest, 2 * k.min(used(plan) - 1), "{cell:?}");
        assert!(
            50 * busiest * POST_NS * layers <= p.step_ns,
            "{cell:?}: {busiest} messages a layer"
        );
    };
    each_cell(RULER, 3_000, check);
    let shape = Shape {
        name: "k1",
        experts: 7,
        used: 1,
        expert: 1_000_000_000,
        ..V41
    };
    let model = shape.model(false);
    let machine = spark(GB10_POOL, shape.layers);
    let terms = terms_of(&model, &machine, shape.n_embd);
    let mut nodes = spark_nodes(4, shape.layers, fabric(4, LAMBDAS[0]));
    let budgets = budgets_on(&model, &mut nodes, &vec![machine; 4]);
    let whole = Workload {
        whole: true,
        ..workload(3_000)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("k1 on four Sparks");
    assert!(plan.owned().iter().all(|&o| o > 0), "{:?}", plan.owned());
    check(("k1", 4, LAMBDAS[0] / 1000, false), &plan);
    let o: Vec<u128> = plan.owned().iter().map(|&x| u128::from(x)).collect();
    let (a, sh, e) = shape.bytes(false);
    let ps = |b: u64| u128::from(b) * PS / u128::from(GB10_READ);
    let lambda = u128::from(LAMBDAS[0] + HOP_NS) * 1000;
    let layer = o[0] * ps(sh + e) + (o[1] + o[2] + o[3]) * ps(sh).max(lambda + ps(e));
    let p = plan.predict().expect("a prediction");
    assert_eq!(
        u128::from(p.step_ns),
        (ps(a) + shape.layers as u128 * layer / 7) / 1000,
        "{:?}",
        plan.owned()
    );
    let busy = nic_of(shape, plan.owned(), p.step_ns, 3_000);
    assert!(
        u128::from(p.per_node[0].nic_permille).abs_diff(busy) <= 1,
        "{} ‰, {busy} ‰",
        p.per_node[0].nic_permille
    );
}

/// Clause 7, `never_slower`: a node is named idle for its gain exactly in
/// §9.8's criterion 7 cells but Qwen3.8 at N = 4, λ 70, A low, where
/// criterion 2 names it first, and no chosen plan is slower than the N − 1
/// plan on the same link: a pair for a ring of three, three switched Sparks
/// for four. The ruler is read from the machine: on one whose ruler is 4 %,
/// Qwen3.8 names node 1 idle at N = 2 where it gains under 8 %, at λ 70.
/// The gain is read on the exact steps: three ids of 5,000 B at top-k 2
/// read at 10¹² B/s take 30,000 ps on one node and 27,777⁷⁄₉ ps on two, a
/// gain of 7.4 %, under a ruler of 5 % twice, and read at 10¹⁴ B/s 300 and
/// 277⁷⁄₉ ps, a gain of 7.41 %, under a ruler of 3.8 % twice.
#[test]
fn never_slower() {
    let gain = |plan: &NodePlan<'_>| {
        plan.idle()
            .iter()
            .any(|i| matches!(i.why, IdleWhy::Gain { .. }))
    };
    let mut fired = Vec::new();
    let mut slower = Vec::new();
    each_cell(RULER, 3_000, |cell, plan| {
        if gain(plan) {
            fired.push(cell);
        }
        let (_, n, lambda, _) = cell;
        if n == 1 {
            return;
        }
        let case = case_of(cell);
        let mut nodes = spark_nodes(n - 1, case.shape.layers, fabric(n - 1, lambda * 1000));
        if n == 4 {
            nodes.fabric = plan.nodes().fabric;
        }
        let budgets = case.budgets(n - 1);
        let step = |p: &NodePlan<'_>| p.step_ps().expect("a step");
        let before = case
            .plan(&nodes, &budgets, plan.workload())
            .expect("the N − 1 plan");
        if step(plan) > step(&before) {
            slower.push((cell, step(plan), step(&before)));
        }
    });
    let qwen = |n, lambda, high| ("Qwen3.8", n, lambda, high);
    assert_eq!(
        fired,
        vec![
            qwen(4, 25, false),
            qwen(4, 25, true),
            qwen(3, 70, true),
            qwen(4, 70, true),
        ]
    );
    assert_eq!(slower, vec![]);
    let mut wide = Vec::new();
    each_cell(4 * RULER, 3_000, |cell, plan| {
        if cell.0 == "Qwen3.8" && cell.1 == 2 && gain(plan) {
            wide.push(cell);
        }
    });
    assert_eq!(wide, vec![qwen(2, 70, false), qwen(2, 70, true)]);
    let model = tiny(3, 2, 5_000);
    let machine = spark(GB10_POOL, 1);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.ruler_permille) = (1_000_000_000_000, 50);
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = terms_of(&model, &machine, 1);
    let whole = Workload {
        whole: true,
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        (plan.owned(), plan.idle()),
        (
            &[3, 0][..],
            &[NodeIdle {
                node: 1,
                why: IdleWhy::Gain {
                    without_ns: 30,
                    with_ns: 27
                }
            }][..]
        )
    );
    (nodes.read, nodes.ruler_permille) = (100_000_000_000_000, 38);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        (plan.owned(), plan.idle()),
        (
            &[3, 0][..],
            &[NodeIdle {
                node: 1,
                why: IdleWhy::Gain {
                    without_ns: 0,
                    with_ns: 0
                }
            }][..]
        )
    );
}

/// C3 (`docs/research/tierengine.md` (k)), `fills_balance`: the split counts
/// each node's fills and pages no id while a pool has room. V4.1 on three
/// Sparks whose remote pools are 60 GB fits the three pools whole at 115 GB
/// beside them and pages none; at 80 and 60 GB the pools hold fewer than
/// every id, the windows with their fills place the paged ids on both
/// remotes, and the remotes' fills are named apart. Fills under one a token
/// are compared whole: five ids at top-k 1 on three Sparks of one id's room
/// each page one id on node 1 and none on node 2, 0.2 and 0 thousandths of
/// a fill a token at a hit rate of 999 ‰, which are apart.
#[test]
fn fills_balance() {
    let case = &cases()[0];
    let remote = routed_only(&V41.model(false));
    let w = Workload {
        fills: Some(Fills {
            hit_permille: 750,
            fill_ns: 430_000,
        }),
        ..workload(3_000)
    };
    let on = |pools: [u64; 2]| {
        let mut nodes: Nodes = spark_nodes(3, V41.layers, fabric(3, LAMBDAS[0]));
        let mut budgets = case.budgets(3);
        for (j, free) in [1, 2].into_iter().zip(pools) {
            let machine = spark(free, V41.layers);
            nodes.machine.nodes[j] = machine.nodes[0].clone();
            budgets[j] = NodeBudget::of(&today(&remote, &machine, false)).expect("a budget");
        }
        let plan = case
            .plan(&nodes, &budgets, w)
            .expect("V4.1 on three Sparks");
        let p = plan.predict().expect("a prediction");
        let fills: Vec<u64> = p.per_node.iter().map(|l| l.fills_milli).collect();
        (fills, plan.fills_apart().expect("a prediction"))
    };
    assert_eq!(on([115_000_000_000, 60_000_000_000]), (vec![0, 0, 0], None));
    assert_eq!(
        on([80_000_000_000, 60_000_000_000]),
        (vec![0, 1093, 3125], Some(FillsApart { low: 1, high: 2 }))
    );
    let id = 1_000_000_000;
    let model = tiny(5, 1, id);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(3, 1, Fabric::Ring { link });
    let machines = with_room(&model, id, &[1, 1, 1]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let w = Workload {
        fills: Some(Fills {
            hit_permille: 999,
            fill_ns: 1,
        }),
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, w).expect("three Sparks");
    assert_eq!(plan.owned(), [2, 2, 1]);
    assert_eq!(
        plan.fills_apart().expect("a prediction"),
        Some(FillsApart { low: 2, high: 1 })
    );
}

/// Undefined inputs and sums past their integers are refused by name: a
/// switch whose round trip and hop pass u64, a token whose slots' outcomes
/// pass u128 (512 experts at top-k 15), a ruler outside 1 to 1000 permille,
/// a prompt rate or row width of 0, today's card plan of a machine of two
/// nodes or of a pool whose bytes with its card's context pass u64, and a
/// budget of another node.
#[test]
fn named_refusals() {
    let case = &cases()[4];
    let budgets = case.budgets(2);
    let link = LinkCost {
        round_trip_ns: u64::MAX,
    };
    let far = spark_nodes(2, QWEN38.layers, Fabric::Switch { link, hop_ns: 1 });
    assert!(matches!(
        case.plan(&far, &budgets, workload(3_000)),
        Err(NodesError::Overflow { .. })
    ));
    let mut nodes = spark_nodes(2, QWEN38.layers, fabric(2, LAMBDAS[0]));
    let wide = Terms {
        experts_used: 15,
        ..case.terms.clone()
    };
    assert!(matches!(
        plan_nodes(&nodes, &wide, &budgets, workload(3_000)),
        Err(NodesError::Overflow {
            experts: 512,
            used: 15
        })
    ));
    let flat = Terms {
        n_embd: 0,
        ..case.terms.clone()
    };
    assert!(matches!(
        plan_nodes(&nodes, &flat, &budgets, workload(3_000)),
        Err(NodesError::Undefined("a row width of 0"))
    ));
    assert!(matches!(
        case.plan(&nodes, &budgets, workload(0)),
        Err(NodesError::Undefined("a prompt rate of 0"))
    ));
    for ruler in [0, 1001, u64::MAX] {
        nodes.ruler_permille = ruler;
        assert!(
            matches!(
                case.plan(&nodes, &budgets, workload(3_000)),
                Err(NodesError::Undefined("a ruler outside 1..=1000 permille"))
            ),
            "ruler {ruler}"
        );
    }
    let model = QWEN38.model(false);
    assert!(matches!(
        plan_with(&model, &nodes.machine, CTX, &Kv(0), None),
        Err(super::PlacementError::Nodes { nodes: 2 })
    ));
    assert!(matches!(
        plan_with(&model, &spark(u64::MAX, QWEN38.layers), CTX, &Kv(0), None),
        Err(super::PlacementError::PoolPastU64 {
            bytes: u64::MAX,
            ..
        })
    ));
    nodes.ruler_permille = RULER;
    assert!(matches!(
        case.plan(&nodes, &case.budgets(1), workload(3_000)),
        Err(NodesError::Budget(1))
    ));
}
