//! Issue #101: a trained creature must never carry its source's creature-level
//! `uuid`.
//!
//! In NEAT-AI the creature-level `uuid` is *content-derived* — a v5 UUID over
//! the neurons (`uuid`, `type`, `bias`, `squash`, `frozen`), the synapses
//! (`fromUUID`, `toUUID`, `weight`, `type`, `frozen`) and `input`.
//! Backpropagation's whole job is to move biases and weights, so the trained
//! creature is a *different* creature and the inherited uuid is a lie.
//! `makeUUID` short-circuits on a present uuid and never recomputes, and
//! `Fitness.calculate` deduplicates its evaluation queue by uuid — so a stale
//! uuid can hand a trained creature a score it never earned.
//!
//! Two things are deliberately *not* identity and must survive untouched:
//! `tags` (excluded from the hash) and per-neuron `uuid` (an *input* to the
//! hash, and the key per-neuron tags are filed under).

use neat_ai_backpropagation::ffi::{TrainAbiRequest, TrainAbiResponse, train};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};

/// The creature every test starts from: a top-level `uuid`, pedigree tags, and
/// per-neuron `uuid`s on an `input-0 → h1 → o1` identity chain.
const TAGGED_SOURCE: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
  "uuid":"3f1c2b6a-0000-5000-8000-000000000001",
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0}
  ],
  "tags":[
    {"name":"name","value":"Tiny"},
    {"name":"lamarck","value":"🦒 leave me alone"}
  ]
}"#;

/// A learnable corpus: `target = 2 × observation`, which the chain above only
/// reaches by moving weights and biases.
fn corpus() -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let mut file = fs::File::create(data.join("0.bin")).unwrap();
    for (observation, target) in [(1.0f32, 2.0f32), (2.0, 4.0), (3.0, 6.0)] {
        file.write_all(&observation.to_le_bytes()).unwrap();
        file.write_all(&target.to_le_bytes()).unwrap();
    }
    drop(file);
    (dir, data)
}

/// Train `TAGGED_SOURCE` through the C ABI entry point, so both write surfaces
/// — `best.json` and `TrainAbiResponse.best_creature_json` — are produced
/// exactly as a foreign caller produces them.
fn train_tagged_source() -> (TempDir, TrainAbiResponse) {
    let (dir, data) = corpus();
    let out = dir.path().join("out");
    let response = train(&TrainAbiRequest {
        creature_json: TAGGED_SOURCE.to_string(),
        training_data: data,
        output_dir: out,
        epochs: 4,
        disable_random_samples: true,
        seed: 1,
        ..TrainAbiRequest::default()
    })
    .expect("a healthy run must succeed");
    (dir, response)
}

fn parse(text: &str) -> Value {
    serde_json::from_str(text).expect("emitted creature must be JSON")
}

/// The regression test. Both surfaces must emit **no** creature-level `uuid`:
/// the consumer derives it from the content it actually received.
#[test]
fn a_trained_creature_carries_no_source_uuid_on_either_surface() {
    let (_dir, response) = train_tagged_source();
    let from_abi = parse(&response.best_creature_json);
    assert!(
        from_abi.get("uuid").is_none(),
        "TrainAbiResponse.bestCreatureJson kept the source uuid: {}",
        response.best_creature_json
    );

    let on_disk_text = fs::read_to_string(&response.best_path).expect("best.json must be written");
    assert_eq!(
        on_disk_text, response.best_creature_json,
        "best.json and the ABI payload must be the same bytes"
    );
    assert!(
        parse(&on_disk_text).get("uuid").is_none(),
        "best.json kept the source uuid: {on_disk_text}"
    );
}

/// Anti-vacuity cross-check: the run really did move the content the uuid
/// hashes over, so the assertion above cannot pass on a no-op run.
#[test]
fn the_trained_creature_differs_structurally_from_its_source() {
    let (_dir, response) = train_tagged_source();
    let trained = parse(&response.best_creature_json);
    let source = parse(TAGGED_SOURCE);
    assert!(response.accepted_epochs > 0, "no epoch was accepted");
    assert_ne!(
        trained["neurons"], source["neurons"],
        "biases did not move — the identity assertion would be vacuous"
    );
    assert_ne!(
        trained["synapses"], source["synapses"],
        "weights did not move — the identity assertion would be vacuous"
    );
}

/// The over-correction guard: dropping the creature-level uuid must not turn
/// into "strip everything". Tags are excluded from the uuid hash, and
/// per-neuron `uuid`s are an *input* to it — both survive byte-identical.
#[test]
fn dropping_the_creature_uuid_keeps_tags_and_per_neuron_uuids() {
    let (_dir, response) = train_tagged_source();
    let trained = parse(&response.best_creature_json);
    let source = parse(TAGGED_SOURCE);

    let tags = trained["tags"].as_array().expect("tags must survive");
    for source_tag in source["tags"].as_array().unwrap() {
        let name = source_tag["name"].as_str().unwrap();
        let kept = tags
            .iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("pedigree tag {name} was dropped"));
        assert_eq!(
            kept["value"], source_tag["value"],
            "tag {name} was rewritten"
        );
    }

    let source_neuron_uuids: Vec<&Value> = source["neurons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| &n["uuid"])
        .collect();
    let trained_neuron_uuids: Vec<&Value> = trained["neurons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| &n["uuid"])
        .collect();
    assert_eq!(
        trained_neuron_uuids, source_neuron_uuids,
        "per-neuron uuids are identity labels and must be preserved"
    );

    let endpoints = |value: &Value| -> Vec<(String, String)> {
        value["synapses"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["fromUUID"].as_str().unwrap().to_string(),
                    s["toUUID"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    };
    assert_eq!(
        endpoints(&trained),
        endpoints(&source),
        "synapse endpoints reference neuron uuids and must be preserved"
    );
}
