import tinytuya

d = tinytuya.BulbDevice(
    "bf0057a91523c3c56bxtc4",
    "192.168.0.211",
    "DIN_LOCAL_KEY"
)

# TESTA dessa en i taget
d.set_version(3.3)
# d.set_version(3.4)
# d.set_version(3.5)

print(d.status())