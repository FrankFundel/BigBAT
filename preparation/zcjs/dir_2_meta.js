var fs = require("fs");
const { parse } = require("csv-parse");

const ZCJS = require("./zcjs.js");

var labelsPath = "G:/UFS DATA 2/labels/";
var metaPath = "G:/UFS DATA 2/meta.csv";
var relabel = {
  Rcap: "Rhinolophus capensis",
  Taphmaur: "Taphozous mauritianus",
  Mnat: "Minopterus natalensis",
  Scotding: "Scotophilus dinganii",
  Vesp52: "Vespertilionidae",
  Cistugo: "Cistugo lesueuri",
  Rsmith: "Rhinolophus smithersi",
  Rclivos: "Rhinolophus clivosus",
  Lcap: "Laephotis capensis",
  Rousaegy: "Rousettus aegyptiacus",
  Phesp: "Pipistrellus hesperidus",
  Lcapbuz: "Laephotis capensis",
  Cpumbuz: "Chaerephon pumilus",
  RSimu: "Rhinolophus simulator",
  Keriv: "Kerivoula",
  Ehott: "Eptesicus hottentotus",
  Cles: "Cistugo lesueuri",
  Cpum: "Chaerephon pumilus",
  Taegbuz: "Tadarida aegyptiaca",
  SprayBuz: "Pesticide Spray",
  Taeg: "Tadarida aegyptiaca",
  Mtric: "Myotis tricolor",
  Mwel: "Myotis welwitschii",
  Phespbuz: "Pipistrellus hesperidus",
};

const result = [];

const getSpec = (zcUrl) => {
  return new Promise((resolve) => {
    fs.exists(zcUrl, function (doesExist) {
      if (doesExist) {
        var p = new ZCJS();
        p.setURL(zcUrl, (data) => {
          let spec = p.anabatHeader().species.trim();
          if (spec == "") {
            resolve("Unlabeled");
          } else {
            let species = [];
            for (let s of spec.split(",")) {
              if (relabel[s]) {
                species.push(relabel[s]);
              }
            }
            species = [...new Set(species)];
            if (species.length == 0) {
              resolve("Unlabeled");
            } else {
              console.log(species.join(","));
              resolve(species.join(","));
            }
          }
        });
      } else {
        resolve("Unlabeled");
      }
    });
  });
};

fs.createReadStream(metaPath)
  .pipe(parse())
  .on("data", async (row) => {
    result.push(row);
  })
  .on("end", async () => {
    // Change
    let i = 0;
    for (let row of result) {
      if (i > 0) {
        let zcUrl = labelsPath + row[2].replace(".wav", "_000.zc");
        row[24] = await getSpec(zcUrl);
      }
      i++;
    }

    // Write
    var csv = "";
    for (let i of result) {
      csv += i.join(";") + "\r\n";
    }
    fs.writeFile("meta_n.csv", csv, "utf8", function (err) {
      if (err) {
        console.log(
          "Some error occured - file either not saved or corrupted file saved."
        );
      } else {
        console.log("It's saved!");
      }
    });
  });
