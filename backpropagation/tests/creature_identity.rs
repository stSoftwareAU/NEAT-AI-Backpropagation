//! Issue #101: a trained creature must never carry the source creature's
//! content-derived `uuid`.
//!
//! NEAT-AI's creature-level `uuid` is a v5 hash over neurons (uuid, type, bias,
//! squash, frozen), synapses (fromUUID, toUUID, weight, type, frozen) and
//! `input`. Training moves every bias and weight, so re-attaching the source
//! uuid publishes a hash that no longer describes the content — and NEAT-AI's
//! `makeUUID` short-circuits on a present uuid, so the lie survives every hop.
//!
//! Both write surfaces are covered here: the bytes on `best.json` and the
//! identical bytes handed back over the C ABI as
//! `TrainAbiResponse.best_creature_json`.

use neat_ai_backpropagation::ffi::{
    NEAT_BACKPROP_OK, NeatBackpropBuffer, TrainAbiResponse, neat_backprop_buffer_free,
    neat_backprop_train,
};
use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Identity chain carrying a creature-level `uuid` and pedigree tags — the
/// exact shape a fleet creature file has when backprop picks it up.
const TAGGED_CHAIN: &str = r#"{
  "uuid":"3f1c2b6a-0000-5000-8000-000000000001",
  "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY",
     "tags":[{"name":"discovery","value":"🔬 scan 42"}]},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY",
     "tags":[{"name":"intelligentDesign","value":"💍 grafted"}]}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0}
  ],
  "tags":[
    {"name":"name","value":"Chain"},
    {"name":"lamarck","value":"🦒 leave me alone"}
  ]
}"#;

/// A one-record corpus mapping `input=1.0` onto `target=2.0`.
fn corpus() -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let mut file = fs::File::create(data.join("0.bin")).unwrap();
    file.write_all(&1.0f32.to_le_bytes()).unwrap();
    file.write_all(&2.0f32.to_le_bytes()).unwrap();
    (dir, data)
}

/// Call the ABI exactly as a foreign caller does and hand back status + payload.
fn call_train(request: &str) -> (i32, String) {
    let bytes = request.as_bytes();
    let mut out = NeatBackpropBuffer {
        data: std::ptr::null_mut(),
        len: 0,
        capacity: 0,
    };
    // SAFETY: `bytes` outlives the call and `out` is a live, writable buffer
    // that owns nothing yet.
    let status = unsafe { neat_backprop_train(bytes.as_ptr(), bytes.len(), &raw mut out) };
    assert!(
        !out.data.is_null(),
        "every call must populate the out buffer"
    );
    // SAFETY: the library reported `out.len` readable bytes at `out.data`.
    let payload = unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec();
    // SAFETY: `out` was produced by this library and has not been freed.
    unsafe { neat_backprop_buffer_free(&raw mut out) };
    (
        status,
        String::from_utf8(payload).expect("payload is UTF-8"),
    )
}

/// Train `TAGGED_CHAIN` on the learnable corpus for enough epochs to accept.
fn train_tagged_chain(data: &Path, out_dir: &Path) -> TrainAbiResponse {
    let request = json!({
        "creatureJson": TAGGED_CHAIN,
        "trainingData": data,
        "outputDir": out_dir,
        "epochs": 4,
        "maxRecords": 1,
        "seed": 7,
        "learningRate": 0.5,
    });
    let (status, payload) = call_train(&request.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");
    serde_json::from_str(&payload).unwrap()
}

/// The defect: `best.json` and the C ABI payload both re-attached the source
/// creature's uuid after training moved every weight and bias.
#[test]
fn a_trained_creature_carries_no_source_uuid_on_either_surface() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let response = train_tagged_chain(&data, &out_dir);
    assert!(
        response.accepted_epochs > 0,
        "a learnable corpus must accept at least one epoch"
    );

    let from_abi: Value = serde_json::from_str(&response.best_creature_json).unwrap();
    assert!(
        from_abi.get("uuid").is_none(),
        "the C ABI payload must not carry a creature-level uuid: {}",
        response.best_creature_json
    );

    let on_disk_text = fs::read_to_string(out_dir.join("best.json")).unwrap();
    let on_disk: Value = serde_json::from_str(&on_disk_text).unwrap();
    assert!(
        on_disk.get("uuid").is_none(),
        "best.json must not carry a creature-level uuid: {on_disk_text}"
    );
    assert_eq!(
        on_disk_text, response.best_creature_json,
        "both surfaces must be the same bytes"
    );
}

/// Guard against a vacuous pass: the run really did change the content the
/// uuid hashes over, so the creature genuinely needs a new identity.
#[test]
fn the_trained_creature_differs_structurally_from_its_source() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let response = train_tagged_chain(&data, &out_dir);

    let source: Value = serde_json::from_str(TAGGED_CHAIN).unwrap();
    let trained: Value = serde_json::from_str(&response.best_creature_json).unwrap();
    assert_ne!(
        source["neurons"], trained["neurons"],
        "training must have moved at least one bias"
    );
    assert_ne!(
        source["synapses"], trained["synapses"],
        "training must have moved at least one weight"
    );
}

/// The over-correction guard: dropping the creature-level uuid must not turn
/// into "strip everything". Pedigree tags survive, and per-neuron `uuid` —
/// a stable identity label that is an *input* to the creature hash — is
/// byte-identical to the source.
#[test]
fn dropping_the_creature_uuid_keeps_tags_and_per_neuron_uuids() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let response = train_tagged_chain(&data, &out_dir);

    let source: Value = serde_json::from_str(TAGGED_CHAIN).unwrap();
    let trained: Value = serde_json::from_str(&response.best_creature_json).unwrap();

    let tags = trained["tags"].as_array().expect("tags are re-attached");
    for expected in source["tags"].as_array().unwrap() {
        let name = expected["name"].as_str().unwrap();
        let found = tags
            .iter()
            .find(|t| t["name"] == expected["name"])
            .unwrap_or_else(|| panic!("pedigree tag {name} was dropped"));
        assert_eq!(
            found["value"], expected["value"],
            "tag {name} was rewritten"
        );
    }

    let neuron_uuids = |value: &Value| -> Vec<String> {
        value["neurons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["uuid"].as_str().unwrap().to_string())
            .collect()
    };
    assert_eq!(
        neuron_uuids(&source),
        neuron_uuids(&trained),
        "per-neuron uuids must be preserved verbatim"
    );

    let synapse_ends = |value: &Value| -> Vec<(String, String)> {
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
        synapse_ends(&source),
        synapse_ends(&trained),
        "synapse endpoints must be preserved verbatim"
    );
}

/// GRQ #4491: per-neuron provenance survives training on both write surfaces.
///
/// `NeuronExport` models no `tags` field, so every neuron's discovery /
/// intelligent-design tag used to be dropped on write. GRQ's #4216 check-in
/// guard refuses a candidate that lost the source's per-neuron tags, so once
/// that guard reached the Backprop worker no trained creature could be
/// published at all.
#[test]
fn per_neuron_provenance_tags_survive_training_on_both_surfaces() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let response = train_tagged_chain(&data, &out_dir);

    let source: Value = serde_json::from_str(TAGGED_CHAIN).unwrap();
    let on_disk_text = fs::read_to_string(out_dir.join("best.json")).unwrap();

    let neuron_tags = |value: &Value| -> Vec<(String, Value)> {
        value["neurons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                (
                    n["uuid"].as_str().unwrap().to_string(),
                    n.get("tags").cloned().unwrap_or(Value::Null),
                )
            })
            .collect()
    };

    let expected = neuron_tags(&source);
    assert!(
        expected.iter().all(|(_, tags)| tags.is_array()),
        "the fixture must actually carry per-neuron tags: {expected:?}"
    );
    for surface in [&response.best_creature_json, &on_disk_text] {
        let trained: Value = serde_json::from_str(surface).unwrap();
        assert_eq!(
            neuron_tags(&trained),
            expected,
            "per-neuron tags must be preserved verbatim: {surface}"
        );
    }
}
