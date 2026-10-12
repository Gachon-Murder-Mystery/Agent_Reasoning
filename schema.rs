//! schema.rs -- Rust port of agent/schema.py, exposed to Python as `schema_rs`.
//!
//! Same surface as the Python version: BeliefState(candidates, agent_id=None),
//! apply, remaining, entropy, total_entropy, settled, evidence_for, summary,
//! to_dict / from_dict / to_json / from_json. agent.py cannot tell the two apart.
//!
//! The reason to do this in Rust is not speed -- the pools hold a few dozen ids.
//! It is that Category is an enum here, so the singular/plural key mix-up that
//! broke _validate cannot compile, and a missing field is an Option the compiler
//! forces us to handle.
//!
//! JSON goes through Python's own json module so the bytes match schema.py
//! exactly; that is what makes the two implementations comparable.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: &str = "b-0.3";

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
    fn index(self) -> usize {
        match self {
            Category::Suspect => 0,
            Category::Weapon => 1,
            Category::Place => 2,
        }
    }
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

#[pyclass(module = "schema_rs")]
pub struct BeliefState {
    #[pyo3(get)]
    schema_version: String,
    #[pyo3(get)]
    agent_id: Option<String>,
    pools: [Pool; 3],
    known_clues: BTreeSet<String>,
    heard_clues: BTreeSet<String>,
}

#[pymethods]
impl BeliefState {
    #[new]
    #[pyo3(signature = (candidates, agent_id = None))]
    fn new(candidates: &Bound<'_, PyDict>, agent_id: Option<String>) -> PyResult<Self> {
        let mut pools: [Pool; 3] = Default::default();
        for category in Category::ALL {
            let pool = &mut pools[category.index()];
            if let Some(entries) = candidates.get_item(category.name())? {
                if entries.is_none() {
                    continue;
                }
                for item in entries.downcast::<PyList>()?.iter() {
                    let (ident, name) = read_entry(&item)?;
                    pool.ids.push(ident);
                    pool.names.insert(ident, name);
                }
            }
        }
        Ok(BeliefState {
            schema_version: SCHEMA_VERSION.to_string(),
            agent_id,
            pools,
            known_clues: BTreeSet::new(),
            heard_clues: BTreeSet::new(),
        })
    }

    /// Fold in one exclusionary clue. Returns whether the belief moved.
    #[pyo3(signature = (clue, source = "own"))]
    fn apply(&mut self, clue: &Bound<'_, PyAny>, source: &str) -> PyResult<bool> {
        let Ok(dict) = clue.downcast::<PyDict>() else {
            return Ok(false);
        };

        let clue_id = match dict.get_item("id")? {
            Some(value) if !value.is_none() => value.extract::<String>().ok(),
            _ => None,
        };
        if let Some(ref ident) = clue_id {
            if source == "own" {
                self.known_clues.insert(ident.clone());
            } else {
                self.heard_clues.insert(ident.clone());
            }
        }

        let category = match dict.get_item("category")? {
            Some(value) => value.extract::<String>().ok().and_then(|s| Category::parse(&s)),
            None => None,
        };
        let Some(category) = category else {
            return Ok(false);
        };

        let excluded = match dict.get_item("excluded_id")? {
            Some(value) if !value.is_none() => value.extract::<i64>().ok(),
            _ => None,
        };
        let Some(excluded) = excluded else {
            return Ok(false);
        };

        let pool = &mut self.pools[category.index()];
        if !pool.ids.contains(&excluded) {
            return Ok(false); // never trust an id the pool does not contain
        }

        let trail = pool.evidence.entry(excluded).or_default();
        if let Some(ident) = clue_id {
            if !trail.contains(&ident) {
                trail.push(ident);
            }
        }
        if pool.eliminated.contains(&excluded) {
            return Ok(false); // already folded in; repeats stay harmless
        }
        pool.eliminated.insert(excluded);
        Ok(true)
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

        let out = PyDict::new_bound(py);
        out.set_item("schema_version", &self.schema_version)?;
        out.set_item("agent_id", &self.agent_id)?;
        out.set_item("pools", pools)?;
        out.set_item("names", names)?;
        out.set_item("eliminated", eliminated)?;
        out.set_item("evidence", evidence)?;
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
        let eliminated = data.get_item("eliminated")?;
        let evidence = data.get_item("evidence")?;

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
            if let Some(ref table) = eliminated {
                if let Some(sub) = table.downcast::<PyDict>()?.get_item(category.name())? {
                    pool.eliminated = sub.extract::<Vec<i64>>()?.into_iter().collect();
                }
            }
            if let Some(ref table) = evidence {
                if let Some(sub) = table.downcast::<PyDict>()?.get_item(category.name())? {
                    for (key, value) in sub.downcast::<PyDict>()?.iter() {
                        let ident: i64 = key.extract::<String>()?.parse().map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err("non-integer candidate id")
                        })?;
                        pool.evidence.insert(ident, value.extract()?);
                    }
                }
            }
        }

        if let Some(value) = data.get_item("known_clues")? {
            state.known_clues = value.extract::<Vec<String>>()?.into_iter().collect();
        }
        if let Some(value) = data.get_item("heard_clues")? {
            state.heard_clues = value.extract::<Vec<String>>()?.into_iter().collect();
        }
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