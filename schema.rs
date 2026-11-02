//! schema.rs -- Rust port of agent/schema.py, exposed to Python as `schema_rs`.
//!
//! Same surface as the Python version: BeliefState(candidates, agent_id=None),
//! apply, infer, remaining, entropy, total_entropy, settled, evidence_for,
//! summary, to_dict / from_dict / to_json / from_json. agent.py cannot tell the
//! two apart.
//!
//! Ruleset v1.3 turned belief from "one clue, one elimination" into a closure
//! over every fact held, because indirect midnight facts only bite in
//! combination. The closure below is a port of the engine's
//! competitive_rules.infer() and must stay rule-for-rule identical to it.
//!
//! The reason to do this in Rust is not speed -- the pools hold a few dozen ids.
//! It is that Category and Fact are enums here, so the singular/plural key
//! mix-up that broke _validate cannot compile, a clue shape we forgot to handle
//! is a non-exhaustive match, and a missing field is an Option the compiler
//! forces us to deal with.
//!
//! JSON goes through Python's own json module so the bytes match schema.py
//! exactly; that is what makes the two implementations comparable.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: &str = "b-0.4";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Category {
    Suspect,
    Weapon,
    Place,
}

impl Category {
    const ALL: [Category; 3] = [Category::Suspect, Category::Weapon, Category::Place];
    fn parse(name: &str) -> Option<Category> {
        match name {
            "suspect" => Some(Category::Suspect),
            "weapon" => Some(Category::Weapon),
            "place" => Some(Category::Place),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Category::Suspect => "suspect",
            Category::Weapon => "weapon",
            Category::Place => "place",
        }
    }
    /// The engine sends pools under either key depending on the view.
    fn aliases(self) -> [&'static str; 2] {
        match self {
            Category::Suspect => ["suspect", "suspects"],
            Category::Weapon => ["weapon", "weapons"],
            Category::Place => ["place", "places"],
        }
    }
    fn index(self) -> usize {
        match self {
            Category::Suspect => 0,
            Category::Weapon => 1,
            Category::Place => 2,
        }
    }
}

const SUSPECT: usize = 0;
const WEAPON: usize = 1;
const PLACE: usize = 2;

/// What a clue actually proves. `Opaque` is a narrated or possession packet:
/// held, counted, never asserted.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Fact {
    Definite { category: Category, candidate: i64 },
    CharacterRoom { character: i64, room: i64 },
    WeaponRoom { weapon: i64, room: i64 },
    Companion { character: i64, other: i64 },
    Opaque,
}

#[derive(Clone, Default)]
struct Pool {
    ids: Vec<i64>,
    names: BTreeMap<i64, String>,
    eliminated: BTreeSet<i64>,
    evidence: BTreeMap<i64, Vec<String>>,
}

impl Pool {
    fn remaining(&self) -> Vec<i64> {
        self.ids
            .iter()
            .copied()
            .filter(|i| !self.eliminated.contains(i))
            .collect()
    }
    fn entropy(&self) -> f64 {
        let n = self.remaining().len();
        if n == 0 {
            0.0
        } else {
            (n as f64).log2()
        }
    }
    fn settled(&self) -> Option<i64> {
        let alive = self.remaining();
        if alive.len() == 1 {
            Some(alive[0])
        } else {
            None
        }
    }
}

fn round_to(value: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (value * factor).round() / factor
}

fn one(clue_id: &str) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    set.insert(clue_id.to_string());
    set
}

/// A candidate entry is either {"id": int, "name": str} or a bare int.
fn read_entry(item: &Bound<'_, PyAny>) -> PyResult<(i64, String)> {
    if let Ok(dict) = item.downcast::<PyDict>() {
        let ident = match dict.get_item("id")? {
            Some(value) => value.extract::<i64>()?,
            None => return Err(pyo3::exceptions::PyValueError::new_err("candidate without id")),
        };
        let name = match dict.get_item("name")? {
            Some(value) => value.extract::<String>().unwrap_or_default(),
            None => String::new(),
        };
        Ok((ident, name))
    } else {
        let ident = item.extract::<i64>()?;
        Ok((ident, ident.to_string()))
    }
}

fn opt_str(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    Ok(match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<String>().ok(),
        _ => None,
    })
}

fn opt_int(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    Ok(match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<i64>().ok(),
        _ => None,
    })
}

/// Engine clue -> Fact. `None` means the packet is not even an opaque one.
fn parse_fact(dict: &Bound<'_, PyDict>) -> PyResult<Option<Fact>> {
    let kind = opt_str(dict, "kind")?.unwrap_or_default();
    if kind == "definite" {
        let category = opt_str(dict, "category")?.and_then(|s| Category::parse(&s));
        // v1.3 names the excluded candidate `candidate`; `excluded_id` is the old
        // field name and is still accepted so legacy fixtures keep working.
        let candidate = match opt_int(dict, "candidate")? {
            Some(value) => Some(value),
            None => opt_int(dict, "excluded_id")?,
        };
        return Ok(match (category, candidate) {
            (Some(category), Some(candidate)) => Some(Fact::Definite { category, candidate }),
            _ => None,
        });
    }
    if kind == "narrated" || kind == "possession" {
        return Ok(Some(Fact::Opaque));
    }
    if kind != "indirect" {
        return Ok(None);
    }
    let shape = opt_str(dict, "type")?.unwrap_or_default();
    let room = opt_int(dict, "room")?;
    Ok(match shape.as_str() {
        "character_room" => match (opt_int(dict, "character")?, room) {
            (Some(character), Some(room)) => Some(Fact::CharacterRoom { character, room }),
            _ => None,
        },
        "weapon_room" => match (opt_int(dict, "weapon")?, room) {
            (Some(weapon), Some(room)) => Some(Fact::WeaponRoom { weapon, room }),
            _ => None,
        },
        "companion" => match (opt_int(dict, "character")?, opt_int(dict, "other")?) {
            (Some(character), Some(other)) => Some(Fact::Companion { character, other }),
            _ => None,
        },
        _ => None,
    })
}

#[pyclass(module = "schema_rs")]
pub struct BeliefState {
    #[pyo3(get)]
    schema_version: String,
    #[pyo3(get)]
    agent_id: Option<String>,
    pools: [Pool; 3],
    facts: BTreeMap<String, Fact>,
    opaque: BTreeSet<String>,
    known_clues: BTreeSet<String>,
    heard_clues: BTreeSet<String>,
}

impl BeliefState {
    /// Port of competitive_rules.infer(). Returns whether the closure moved.
    fn close(&mut self) -> bool {
        let mut excluded: [BTreeMap<i64, BTreeSet<String>>; 3] = Default::default();
        let mut positions: [BTreeMap<i64, (i64, BTreeSet<String>)>; 2] = Default::default();
        let mut companions: Vec<(i64, i64, BTreeSet<String>)> = Vec::new();

        for (clue_id, fact) in &self.facts {
            match fact {
                Fact::Definite {
                    category,
                    candidate,
                } => {
                    if self.pools[category.index()].ids.contains(candidate) {
                        excluded[category.index()].insert(*candidate, one(clue_id));
                    }
                }
                Fact::CharacterRoom { character, room } => {
                    positions[SUSPECT].insert(*character, (*room, one(clue_id)));
                }
                Fact::WeaponRoom { weapon, room } => {
                    positions[WEAPON].insert(*weapon, (*room, one(clue_id)));
                }
                Fact::Companion { character, other } => {
                    companions.push((*character, *other, one(clue_id)))
                }
                Fact::Opaque => {}
            }
        }

        let rooms = self.pools[PLACE].ids.clone();
        let mut changed = true;
        while changed {
            changed = false;

            // Someone seen with a located person was in that same room.
            for (a, b, support) in &companions {
                for (x, y) in [(*a, *b), (*b, *a)] {
                    if positions[SUSPECT].contains_key(&x) {
                        continue;
                    }
                    if let Some((room, basis)) = positions[SUSPECT].get(&y).cloned() {
                        let mut merged = support.clone();
                        merged.extend(basis);
                        positions[SUSPECT].insert(x, (room, merged));
                        changed = true;
                    }
                }
            }

            let scene: Vec<i64> = rooms
                .iter()
                .copied()
                .filter(|i| !excluded[PLACE].contains_key(i))
                .collect();
            let place_union: BTreeSet<String> =
                excluded[PLACE].values().flatten().cloned().collect();

            for cat in [SUSPECT, WEAPON] {
                let located: Vec<(i64, (i64, BTreeSet<String>))> = positions[cat]
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .collect();
                for (i, (room, support)) in located {
                    if !self.pools[cat].ids.contains(&i) || excluded[cat].contains_key(&i) {
                        continue;
                    }
                    let basis = match excluded[PLACE].get(&room) {
                        Some(found) => Some(found.clone()),
                        // Only one room can be the scene, and this is not it.
                        None if scene.len() == 1 && scene[0] != room => Some(place_union.clone()),
                        None => None,
                    };
                    if let Some(basis) = basis {
                        let mut merged = basis;
                        merged.extend(support);
                        excluded[cat].insert(i, merged);
                        changed = true;
                    }
                }

                // The one survivor must be the culprit/weapon, so the scene is
                // wherever it was at midnight, and every other room is out.
                let alive: Vec<i64> = self.pools[cat]
                    .ids
                    .iter()
                    .copied()
                    .filter(|i| !excluded[cat].contains_key(i))
                    .collect();
                if alive.len() != 1 || excluded[cat].is_empty() {
                    continue;
                }
                if let Some((room, support)) = positions[cat].get(&alive[0]).cloned() {
                    let mut merged = support;
                    for trail in excluded[cat].values() {
                        merged.extend(trail.iter().cloned());
                    }
                    for i in rooms.iter().copied() {
                        if i != room && !excluded[PLACE].contains_key(&i) {
                            excluded[PLACE].insert(i, merged.clone());
                            changed = true;
                        }
                    }
                }
            }
        }

        let mut moved = false;
        for category in Category::ALL {
            let index = category.index();
            let pool = &mut self.pools[index];
            let killed: BTreeSet<i64> = excluded[index].keys().copied().collect();
            if killed != pool.eliminated {
                moved = true;
            }
            pool.eliminated = killed;
            pool.evidence = excluded[index]
                .iter()
                .map(|(k, v)| (*k, v.iter().cloned().collect()))
                .collect();
        }
        moved
    }
}

#[pymethods]
impl BeliefState {
    #[new]
    #[pyo3(signature = (candidates, agent_id = None))]
    fn new(candidates: &Bound<'_, PyDict>, agent_id: Option<String>) -> PyResult<Self> {
        let mut pools: [Pool; 3] = Default::default();
        for category in Category::ALL {
            let pool = &mut pools[category.index()];
            for key in category.aliases() {
                let Some(entries) = candidates.get_item(key)? else {
                    continue;
                };
                if entries.is_none() {
                    continue;
                }
                let entries = entries.downcast::<PyList>()?;
                if entries.is_empty() {
                    continue;
                }
                for item in entries.iter() {
                    let (ident, name) = read_entry(&item)?;
                    pool.ids.push(ident);
                    pool.names.insert(ident, name);
                }
                break;
            }
        }
        Ok(BeliefState {
            schema_version: SCHEMA_VERSION.to_string(),
            agent_id,
            pools,
            facts: BTreeMap::new(),
            opaque: BTreeSet::new(),
            known_clues: BTreeSet::new(),
            heard_clues: BTreeSet::new(),
        })
    }

    /// Fold in one clue and reclose. Returns whether the belief moved.
    #[pyo3(signature = (clue, source = "own"))]
    fn apply(&mut self, clue: &Bound<'_, PyAny>, source: &str) -> PyResult<bool> {
        let Ok(dict) = clue.downcast::<PyDict>() else {
            return Ok(false);
        };
        let Some(clue_id) = opt_str(dict, "id")? else {
            return Ok(false);
        };
        if source == "own" {
            self.known_clues.insert(clue_id.clone());
        } else {
            self.heard_clues.insert(clue_id.clone());
        }

        let fact = match parse_fact(dict)? {
            Some(Fact::Opaque) => {
                self.opaque.insert(clue_id);
                return Ok(false);
            }
            Some(fact) => fact,
            None => return Ok(false),
        };
        // A narrated packet can later arrive in full, so a real fact always wins.
        self.opaque.remove(&clue_id);
        if self.facts.get(&clue_id) == Some(&fact) {
            return Ok(false); // repeats stay harmless
        }
        self.facts.insert(clue_id, fact);
        Ok(self.close())
    }

    /// Recompute the closure over every fact held.
    fn infer(&mut self) -> bool {
        self.close()
    }

    fn remaining(&self, category: &str) -> PyResult<Vec<i64>> {
        let category = Category::parse(category)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("unknown category"))?;
        Ok(self.pools[category.index()].remaining())
    }

    fn entropy(&self, category: &str) -> PyResult<f64> {
        let category = Category::parse(category)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("unknown category"))?;
        Ok(self.pools[category.index()].entropy())
    }

    fn total_entropy(&self) -> f64 {
        Category::ALL
            .iter()
            .map(|c| self.pools[c.index()].entropy())
            .sum()
    }

    fn settled(&self, category: &str) -> PyResult<Option<i64>> {
        let category = Category::parse(category)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("unknown category"))?;
        Ok(self.pools[category.index()].settled())
    }

    fn evidence_for(&self, category: &str, ident: i64) -> PyResult<Vec<String>> {
        let category = Category::parse(category)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("unknown category"))?;
        Ok(self.pools[category.index()]
            .evidence
            .get(&ident)
            .cloned()
            .unwrap_or_default())
    }

    /// Goes straight into the prompt, so keep it small and readable.
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new_bound(py);
        for category in Category::ALL {
            let pool = &self.pools[category.index()];
            let alive = pool.remaining();
            let share = if alive.is_empty() {
                0.0
            } else {
                round_to(1.0 / alive.len() as f64, 4)
            };

            let listing = PyList::empty_bound(py);
            for ident in &alive {
                let entry = PyDict::new_bound(py);
                entry.set_item("id", ident)?;
                entry.set_item("name", pool.names.get(ident))?;
                entry.set_item("p", share)?;
                listing.append(entry)?;
            }

            let block = PyDict::new_bound(py);
            block.set_item("remaining", listing)?;
            block.set_item("remaining_count", alive.len())?;
            block.set_item("excluded_count", pool.eliminated.len())?;
            block.set_item("entropy", round_to(pool.entropy(), 3))?;
            block.set_item("settled", pool.settled())?;
            out.set_item(category.name(), block)?;
        }
        out.set_item("total_entropy", round_to(self.total_entropy(), 3))?;

        // Indirect facts that have not bitten yet are the reason to keep asking,
        // so the model is told how many it is sitting on.
        let (mut char_rooms, mut weapon_rooms, mut companions, mut definite) = (0, 0, 0, 0);
        for fact in self.facts.values() {
            match fact {
                Fact::Definite { .. } => definite += 1,
                Fact::CharacterRoom { .. } => char_rooms += 1,
                Fact::WeaponRoom { .. } => weapon_rooms += 1,
                Fact::Companion { .. } => companions += 1,
                Fact::Opaque => {}
            }
        }
        let midnight = PyDict::new_bound(py);
        midnight.set_item("character_rooms", char_rooms)?;
        midnight.set_item("weapon_rooms", weapon_rooms)?;
        midnight.set_item("companions", companions)?;
        out.set_item("midnight_facts", midnight)?;
        out.set_item("definite_count", definite)?;
        out.set_item("unusable_count", self.opaque.len())?;
        Ok(out)
    }

    // ---------- serialisation (for C's takeover / forced termination) ----------

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let pools = PyDict::new_bound(py);
        let names = PyDict::new_bound(py);
        let eliminated = PyDict::new_bound(py);
        let evidence = PyDict::new_bound(py);

        for category in Category::ALL {
            let pool = &self.pools[category.index()];
            pools.set_item(category.name(), pool.ids.clone())?;

            let name_map = PyDict::new_bound(py);
            for (ident, name) in &pool.names {
                name_map.set_item(ident.to_string(), name)?;
            }
            names.set_item(category.name(), name_map)?;

            eliminated.set_item(
                category.name(),
                pool.eliminated.iter().copied().collect::<Vec<i64>>(),
            )?;

            let trail_map = PyDict::new_bound(py);
            for (ident, trail) in &pool.evidence {
                trail_map.set_item(ident.to_string(), trail.clone())?;
            }
            evidence.set_item(category.name(), trail_map)?;
        }

        // Facts use schema.py's wire shape so either side can read the other.
        let facts = PyDict::new_bound(py);
        for (clue_id, fact) in &self.facts {
            let entry = PyDict::new_bound(py);
            match fact {
                Fact::Definite {
                    category,
                    candidate,
                } => {
                    entry.set_item("t", "definite")?;
                    entry.set_item("category", category.name())?;
                    entry.set_item("candidate", candidate)?;
                }
                Fact::CharacterRoom { character, room } => {
                    entry.set_item("t", "character_room")?;
                    entry.set_item("character", character)?;
                    entry.set_item("room", room)?;
                }
                Fact::WeaponRoom { weapon, room } => {
                    entry.set_item("t", "weapon_room")?;
                    entry.set_item("weapon", weapon)?;
                    entry.set_item("room", room)?;
                }
                Fact::Companion { character, other } => {
                    entry.set_item("t", "companion")?;
                    entry.set_item("character", character)?;
                    entry.set_item("other", other)?;
                }
                Fact::Opaque => continue,
            }
            facts.set_item(clue_id, entry)?;
        }

        let out = PyDict::new_bound(py);
        out.set_item("schema_version", &self.schema_version)?;
        out.set_item("agent_id", &self.agent_id)?;
        out.set_item("pools", pools)?;
        out.set_item("names", names)?;
        out.set_item("eliminated", eliminated)?;
        out.set_item("evidence", evidence)?;
        out.set_item("facts", facts)?;
        out.set_item(
            "opaque",
            self.opaque.iter().cloned().collect::<Vec<String>>(),
        )?;
        out.set_item(
            "known_clues",
            self.known_clues.iter().cloned().collect::<Vec<String>>(),
        )?;
        out.set_item(
            "heard_clues",
            self.heard_clues.iter().cloned().collect::<Vec<String>>(),
        )?;
        Ok(out)
    }

    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let py = data.py();
        let empty = PyDict::new_bound(py);
        let agent_id = match data.get_item("agent_id")? {
            Some(value) if !value.is_none() => Some(value.extract::<String>()?),
            _ => None,
        };
        let mut state = BeliefState::new(&empty, agent_id)?;

        if let Some(value) = data.get_item("schema_version")? {
            state.schema_version = value.extract()?;
        }

        let pools = data
            .get_item("pools")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("pools"))?;
        let pools = pools.downcast::<PyDict>()?;
        let names = data.get_item("names")?;

        for category in Category::ALL {
            let pool = &mut state.pools[category.index()];
            if let Some(ids) = pools.get_item(category.name())? {
                pool.ids = ids.extract()?;
            }
            if let Some(ref table) = names {
                if let Some(sub) = table.downcast::<PyDict>()?.get_item(category.name())? {
                    for (key, value) in sub.downcast::<PyDict>()?.iter() {
                        let ident: i64 = key.extract::<String>()?.parse().map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err("non-integer candidate id")
                        })?;
                        pool.names.insert(ident, value.extract()?);
                    }
                }
            }
        }

        if let Some(facts) = data.get_item("facts")? {
            for (key, value) in facts.downcast::<PyDict>()?.iter() {
                let clue_id: String = key.extract()?;
                let entry = value.downcast::<PyDict>()?;
                let shape = opt_str(entry, "t")?.unwrap_or_default();
                let fact = match shape.as_str() {
                    "definite" => {
                        let category = opt_str(entry, "category")?
                            .and_then(|s| Category::parse(&s))
                            .ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err("unknown category")
                            })?;
                        Fact::Definite {
                            category,
                            candidate: opt_int(entry, "candidate")?.unwrap_or_default(),
                        }
                    }
                    "character_room" => Fact::CharacterRoom {
                        character: opt_int(entry, "character")?.unwrap_or_default(),
                        room: opt_int(entry, "room")?.unwrap_or_default(),
                    },
                    "weapon_room" => Fact::WeaponRoom {
                        weapon: opt_int(entry, "weapon")?.unwrap_or_default(),
                        room: opt_int(entry, "room")?.unwrap_or_default(),
                    },
                    "companion" => Fact::Companion {
                        character: opt_int(entry, "character")?.unwrap_or_default(),
                        other: opt_int(entry, "other")?.unwrap_or_default(),
                    },
                    _ => continue,
                };
                state.facts.insert(clue_id, fact);
            }
        }
        if let Some(value) = data.get_item("opaque")? {
            state.opaque = value.extract::<Vec<String>>()?.into_iter().collect();
        }
        if let Some(value) = data.get_item("known_clues")? {
            state.known_clues = value.extract::<Vec<String>>()?.into_iter().collect();
        }
        if let Some(value) = data.get_item("heard_clues")? {
            state.heard_clues = value.extract::<Vec<String>>()?.into_iter().collect();
        }
        // Derived from the facts, so recompute rather than trust the payload.
        state.close();
        Ok(state)
    }

    /// Python's json module, so the bytes match schema.py exactly.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        let json = py.import_bound("json")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("ensure_ascii", false)?;
        kwargs.set_item("sort_keys", true)?;
        json.call_method("dumps", (self.to_dict(py)?,), Some(&kwargs))?
            .extract()
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, text: &str) -> PyResult<Self> {
        let json = py.import_bound("json")?;
        let data = json.call_method1("loads", (text,))?;
        BeliefState::from_dict(data.downcast::<PyDict>()?)
    }
}

#[pymodule]
fn schema_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("SCHEMA_VERSION", SCHEMA_VERSION)?;
    module.add("CATEGORIES", ("suspect", "weapon", "place"))?;
    module.add_class::<BeliefState>()?;
    Ok(())
}
